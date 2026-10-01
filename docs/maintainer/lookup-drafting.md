# Multi-source lookup drafting

NInfer supports exact copy drafting alongside MTP, DFlash and DFlash2. The legacy nearest-hit
lookup remains available, while the opt-in voting path adapts TandemLLM's counted suffix-memory
design to NInfer's chain verifier.

Lookup never licenses a token by itself. It supplies a proposal and the target model verifies it.
A bad lookup can reduce throughput; it cannot change the target distribution when speculative
acceptance is configured correctly.

## A/B modes

The existing behavior is the baseline:

```text
--lookup-ngram 0
```

Legacy recent-match lookup remains:

```text
--spec mtp --draft-tokens 3
--lookup-ngram 8
--lookup-strategy recent
```

Counted multi-source lookup without DFlash takeover can also feed MTP:

```text
--lookup-ngram 5
--lookup-strategy vote
```

DFlash/DFlash2 expose three explicit modes:

| mode | neural proposal | lookup chain | purpose |
|---|---|---|---|
| `off` | yes | no takeover | baseline |
| `replace` | yes | replaces the proposed chain on a confident hit | isolate proposal quality |
| `skip` | skipped on an all-lane confident hit | verified directly | measure drafter-cost removal |

Example:

```bash
ninfer-serve MODEL.ninfer \
  --spec dflash2 --draft-tokens 15 \
  --lookup-ngram 5 --lookup-strategy vote \
  --lookup-dflash skip \
  --lookup-min-support 2 --lookup-min-confidence 0.70 \
  --lookup-base-drafts 7 --lookup-deep-after 2 --lookup-deep-drafts 15
```

All lookup additions are opt-in. `recent` remains the strategy default and
`--lookup-dflash off` remains the DFlash default.

## Sources and voting

The voting drafter queries three sources:

1. the current request ledger;
2. a bounded process-lifetime suffix store populated only by successfully finished requests;
3. an optional static corpus suffix array.

Only sources matching the longest available n-gram order vote. Local and process-history
occurrences have weight 1. Static-corpus votes are scaled by `--lookup-corpus-weight`. The best
exact continuation must satisfy both `--lookup-min-support` and
`--lookup-min-confidence`.

This differs intentionally from the old lookup implementation, where the most recent occurrence
won unconditionally.

### Process history

`--lookup-persistent-tokens N` retains at most N token ids across requests. With no path it is
process-local, preserving the original A/B behavior. Add:

```bash
--lookup-persistent-path /var/lib/ninfer/lookup-history.bin
```

to restore/save the bounded token history across clean restarts. The index is rebuilt from the
bounded token snapshot at startup. This remains a proposal hint only: it does **not** replace the
exact context cache and it is never model-state authority. If the bounded store rolls over, older
history is intentionally discarded rather than increasing host memory without limit.

### Static corpus

Build a restart-persistent corpus with:

```bash
python tools/build_lookup_corpus.py \
  --tokenizer /path/to/qwen-tokenizer \
  --src /path/to/code \
  --src /path/to/docs \
  --out-prefix /data/qwen_lookup
```

The builder writes:

```text
/data/qwen_lookup.tokens.i32
/data/qwen_lookup.suffix.u32
/data/qwen_lookup.meta.json
```

Load it with `--lookup-corpus-prefix /data/qwen_lookup`. The suffix array is ordered only to
`max_order`, matching the bounded comparison performed by the runtime.

The corpus format is intentionally simple raw little-endian token/suffix arrays. A corpus must be
built with the same tokenizer as the target artifact; the metadata records the tokenizer hash for
operator verification.

## Deep copy

Copy drafting has no neural block-width requirement. After
`--lookup-deep-after N` consecutive lookup rounds accept their complete proposal, the next lookup
may request `--lookup-deep-drafts K`. Both values are policy maxima: the runtime still clamps K
to the startup DFlash width, output budget and remaining context.

This makes a single experiment configuration usable with both K7 and K15 runtimes.

## DFlash2 sampling correctness

DFlash2 normally uses sparse speculative rejection sampling with its 16-candidate proposal
distribution. A lookup chain is an exact one-hot proposal, q=1.

On a DFlash2 lookup takeover NInfer therefore rewrites the live sparse proposal to:

```text
candidate[0] = lookup_token
q[0]         = 1
q[1..15]     = 0
```

and keeps the existing sparse accept/commit path. This is important for two reasons:

- positive-temperature requests retain exact speculative rejection semantics;
- presence/frequency token counts are published only for the final committed prefix, including a
  Frontend-truncated terminal prefix.

DFlash (non-DFlash2) already uses the dense one-hot proposal contract and needs no sparse rewrite.

## Batch behavior

DFlash takeover currently requires every row in the compact decode batch to have a confident
lookup proposal. Otherwise the entire batch uses the normal neural DFlash path. This keeps one
transaction shape and avoids splitting/reordering active lanes merely for a speculative hint.

Lookup takeover is currently executed eagerly rather than through the fixed neural CUDA Graph,
because its host-selected chain is request data. The fixed/no-hit path remains graphed. Measure
this cost on the target 4090 rather than assuming that head-skip is always profitable.

## Parameters

| flag | default | role |
|---|---:|---|
| `--lookup-ngram N` | 0 | minimum suffix order; 0 disables lookup |
| `--lookup-strategy recent\|vote` | recent | legacy nearest hit or counted multi-source vote |
| `--lookup-dflash off\|replace\|skip` | off | DFlash takeover policy |
| `--lookup-max-order N` | 8 | highest order considered by vote mode |
| `--lookup-max-matches N` | 64 | local/process occurrence cap |
| `--lookup-min-support N` | 1 | minimum winning continuation observations |
| `--lookup-min-confidence F` | 0.60 | minimum winning vote share |
| `--lookup-base-drafts N` | 7 | ordinary copy depth |
| `--lookup-deep-after N` | 2 | consecutive full accepts before deep copy; 0 disables |
| `--lookup-deep-drafts N` | 15 | deep copy policy maximum |
| `--lookup-persistent-tokens N` | 0 | process-history token budget |
| `--lookup-persistent-path PATH` | empty | optional cross-restart bounded history snapshot |
| `--lookup-corpus-prefix PATH` | empty | static corpus prefix |
| `--lookup-corpus-weight F` | 0.50 | static-corpus vote weight |
| `--lookup-corpus-samples N` | 64 | sampled matching suffixes |

## Measurement

Use `scripts/sweeps/dflash2-lookup-realtext.ps1` on model-generated text. Compare at least:

- DFlash2 only;
- vote + replace;
- vote + skip;
- vote + skip + deep-copy.

Record output hashes as well as throughput. Speculative decoding can alter sampled output when its
proposal distribution differs; a speed result without the output and sampling configuration is not
a controlled comparison.

This subsystem is adapted from the counted suffix-memory/corpus ideas in
[0xBakeer/TandemLLM](https://github.com/0xBakeer/TandemLLM). NInfer retains its own paged KV,
ReplaySSM, exact context-cache and transaction machinery.
