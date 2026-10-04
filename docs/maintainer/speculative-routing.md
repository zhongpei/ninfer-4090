# Speculative routing

Routing defaults to `fixed`. `calibrated` uses an offline measured action table;
`stair` learns acceptance-based licensed extents. The modes have separate
contracts and cannot be combined.

## Calibrated DFlash2 chain

Enable a profile explicitly on either CLI or server:

```bash
ninfer-serve MODEL.ninfer \
  --spec dflash2 --draft-tokens 15 \
  --spec-router calibrated --spec-router-profile PATH \
  --greedy --presence-penalty 0 --frequency-penalty 0
```

The initial calibrated route supports DFlash2 chain with a startup K15 drafter,
greedy sampling and zero presence/frequency penalties. If any active request
uses other sampling settings, the entire compact batch uses existing fixed K15
behavior for that round. Tree, lookup and Stair options cannot be combined with
calibrated routing. A profile path is required only for this mode.

Set penalties explicitly when selecting this route: `--greedy` overrides only
temperature, and non-thinking chat can otherwise inherit presence penalty 1.5.
For HTTP requests, specify `temperature: 0`, `presence_penalty: 0` and
`frequency_penalty: 0` or use the corresponding process overrides. Eligibility
uses resolved sampling values.

After the previous round has completely committed, Program selects one action
for the actual compact batch. The table key is active batch 1–8 and the maximum
execution frontier across its lanes: ≤1024, 1025–8192 or 8193–32768. Client
concurrency is not the table key. Missing cells and out-of-domain frontiers
select target-only.

Actions are target-only or K7/K11/K15 with physical target widths 1/8/12/16.
The maximum drafter remains resident. Target-only skips its neural proposal;
speculative actions retain the maximum proposal width. Committed target
features and DFlash context must be caught up before returning to speculation.
Pending replay, KV and recurrent state use the preceding round's actual width,
including when the next action is narrower. Cache restore, terminal commits
and later continuation preserve the same committed prefix.

### Profile contract

Profiles are local JSON artifacts with `schema_version: 1`,
`artifact_type: "ninfer_spec_router_profile"`, `identity`, `cells` and optional
`provenance`. Each cell contains `active_batch`, `frontier_upper`
(1024/8192/32768) and `draft_tokens` (0/7/11/15). Repeated or invalid cells,
unknown fields and malformed JSON are errors.

Identity binds the model artifact ID and prefill signature, hardware class,
backend, KV format, proposal head, startup draft count, CUDA Graph mode,
concurrency, context, prefill chunk, resolved KV capacity, context-cache budgets
and effective execution options. Startup rejects a missing profile or any
identity mismatch. Copying a profile from another artifact or startup
configuration is not a fallback mechanism.

Performance qualification must measure actual resident-K15 control actions,
including target-only's feature/context maintenance. Standalone fixed K7/K11
measurements and Stair masking do not establish these costs. Retain AB/BA pairs
with exact greedy outputs and complete responses. Every retained pair must
beat target-only throughput; median improvement must be at least 2%, and each
pair's request p95 regression must not exceed 5%. Select the highest qualified
end-to-end throughput; actions within 1% of the highest throughput prefer the
smaller K. No qualified benefit or coverage selects target-only.

The profile generator attributes an end-to-end comparison to an actual cell
only when that same cell accounts for at least 95% of full round elapsed time
in every arm. It records minority join/drain cells without declaring coverage
for them. Candidates must cover all represented workloads in the cell; neither
client concurrency nor a minority observed round can fabricate coverage.
Measurement controls may use a forced action profile, but are not qualified
performance profiles.

Graph resource evidence includes prepared definition/executable counts and
`cuda_graph_prepare_peak_device_delta_bytes` / `cuda_graph_prepare_device_delta_bytes`
from `MemorySummary`. These are startup device-wide free-memory deltas, measured
through preparation and first launches, rather than exact Program-owned allocations.
Compare the peak with `cuda_graph_allowance_bytes`; record other device users when
interpreting it. Graph-disabled execution reports zero for these preparation fields.

### Counters

Calibrated logs expose `calibrated_target_only_rounds`, `calibrated_k7_rounds`,
`calibrated_k11_rounds`, `calibrated_k15_rounds`, `calibrated_route_switches`
and `calibrated_fixed_fallback_rounds`. Actions count successfully settled
rounds; fallback rounds also count as K15 actions. Switches compare successive
settled actions, excluding the first; request/cache boundaries do not reset
that history. Engine snapshots are cumulative, not request or interval deltas.
The CLI machine summary labels its scope `published_engine_snapshot`.

## Stair

NInfer can keep a DFlash/DFlash2 drafter at one startup maximum width while choosing a smaller
target-verification extent for each round. Stair changes licensed extent while
retaining the startup physical proposal and target frame widths; it does not skip
the neural drafter or provide the physical target-only route required by calibrated
routing. This is an opt-in A/B feature:

```bash
ninfer-serve MODEL.ninfer \
  --spec dflash2 --draft-tokens 15 \
  --spec-router stair \
  --spec-stair-widths 3,7,11,15 \
  --spec-stair-costs 1.00,1.02,1.05,1.10 \
  --spec-stair-draft-cost 0.25 \
  --spec-router-scope engine \
  --spec-router-state profiles/qwen38-4090.state
```

Omitting `--spec-router` selects `fixed` and retains maximum-K behavior.

## Lifetime and persistence

The original A/B behavior is request-local:

```text
--spec-router-scope request
```

To let serving traffic teach later requests, use:

```bash
--spec-router stair \
--spec-router-scope engine
```

Chain Stair and 16-row Tree-Stair then share Engine lifetime rather than resetting for every
request. To keep those small host-only counters across clean restarts, also set:

```bash
--spec-router-state /var/lib/ninfer/spec-router.state
```

The state contains only observed acceptance/committed-token counters and the last selected rung. It
is an optimization hint, not model/KV/StateImage authority; failure to save during noexcept
destruction never invalidates generated output. The default remains request-local for controlled
A/B comparisons.


## Why the policy is a staircase

Speculative verification is a small-M target forward. Its cost is not generally linear in the
number of verified rows: kernel tile boundaries, split policies and CUDA Graph topology can make
several widths cost nearly the same. A wider block is then attractive only when its expected extra
accepted tokens cost less than the next verify step.

This implementation is derived from the StairCut scheduling idea in
[0xBakeer/TandemLLM](https://github.com/0xBakeer/TandemLLM), reviewed at
`c11aecaa7ba767642409ff51d90d48c79c572d3a`. NInfer keeps its own ReplaySSM, paged-KV,
DFlash context and CUDA Graph machinery; only the scheduling policy is adapted.

## Policy

For a candidate width (K), the router estimates

```text
score(K) = E[committed target tokens | K] / (draft_cost + verify_cost[K])
```

The always-produced target correction/bonus contributes one token. For each draft position,
survival probability is estimated from the request's own attempted and accepted prefix counts,
with a Beta-like prior controlled by `prior_acceptance` and `prior_weight`.

The router deliberately runs the widest configured rung during warmup and periodically afterwards.
Those probes are required: once a policy cuts to a narrow width, later positions are censored and
cannot teach the router whether a wide block would now be profitable.

`switch_margin` retains the previous rung when its score is close to the current best, reducing
oscillation from short-window noise.

## Parameters

| CLI flag | Meaning | Default |
|---|---|---:|
| `--spec-router fixed\|stair\|calibrated` | routing mode | `fixed` |
| `--spec-router-profile PATH` | calibrated identity-bound profile | unset |
| `--spec-router-scope request\|engine` | statistics lifetime | `request` |
| `--spec-router-state PATH` | optional engine-state snapshot | unset |
| `--spec-stair-widths A,B,C,D` | four strictly increasing target draft extents | `3,7,11,15` |
| `--spec-stair-costs A,B,C,D` | positive verify costs in any common relative unit | `1,1.02,1.05,1.10` |
| `--spec-stair-draft-cost F` | one wide DFlash proposal cost in the same unit | `0.25` |
| `--spec-stair-prior F` | initial survival probability | `0.70` |
| `--spec-stair-prior-weight F` | pseudo-observation weight | `2` |
| `--spec-stair-warmup N` | initial widest-rung rounds | `4` |
| `--spec-stair-probe-period N` | wide probe cadence; zero disables | `16` |
| `--spec-stair-margin F` | fractional hysteresis around previous rung | `0.02` |

Stair mode currently applies only to DFlash/DFlash2. Every configured width must be at or below
`--draft-tokens`.

## Stair cost calibration on RTX 4090

Do not copy a DGX Spark or RTX 3090 cost table. Measure the actual sm_89 path at the target context
depths and KV format. The supplied `scripts/sweeps/dflash2-stair-router-realtext.ps1` compares the
fixed baseline with the adaptive policy on model-generated text and records output hashes,
acceptance, tokens/round and decode throughput.

For each intended context class, measure at least the configured rungs and convert the median target
round times to one common relative scale. A masked Stair extent is not evidence of a smaller
physical target width. A/B comparisons should hold model artifact, prompt, KV
format, sampling and CUDA Graph mode constant.

For the K15/24 GB path, `scripts/sweeps/dflash2-stair-cost-calibration.ps1` holds the neural
DFlash2 proposal at K15 and forces each 3/7/11/15 target rung independently. Feed its CSV to:

```bash
python -m tools.dflash2_training.calibrate_4090 sweep.csv --out profiles/4090-stair.json
```

The calibrator uses median total seconds/round as the effective routing cost and therefore emits
`--spec-stair-draft-cost 0` so the common wide-drafter cost is not counted twice. It also refuses
to emit a profile if greedy output hashes differ across forced rungs.

With `--spec-router-scope engine --spec-router-state PATH`, chain and tree counters are loaded
at startup. Each request snapshots the corresponding counters at admission and learns locally.
Only the statistics added by a successfully finished request are merged into the Engine state;
concurrent requests do not alter each other’s current decisions. The snapshot is advisory
performance state; a save failure does not turn a completed generation into a failure.
