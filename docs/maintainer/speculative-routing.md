# Adaptive speculative routing (Stair)

NInfer can keep a DFlash/DFlash2 drafter at one startup maximum width while choosing a smaller
target-verification extent for each round. This is an opt-in A/B feature:

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

Without `--spec-router stair`, routing is `fixed` and the previous maximum-K behavior is
unchanged.

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
| `--spec-router fixed\|stair` | routing mode | `fixed` |
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

## RTX 4090 calibration

Do not copy a DGX Spark or RTX 3090 cost table. Measure the actual sm_89 path at the target context
depths and KV format. The supplied `scripts/sweeps/dflash2-stair-router-realtext.ps1` compares the
fixed baseline with the adaptive policy on model-generated text and records output hashes,
acceptance, tokens/round and decode throughput.

For each intended context class, measure at least the configured rungs and convert the median target
verify times to one common relative scale. A/B comparisons should hold model artifact, prompt, KV
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
