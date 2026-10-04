# DFlash2 selected-width proposal computation

Status: **experimental, disabled by default, no performance or correctness qualification claimed by this change**.
Baseline: `b49d88178c2150e159dbe19687c651f54340ea2c`.

This work separates three questions that must not be conflated:

1. Does physically computing a smaller drafter proposal help at the **same selected K**?
2. Does the independently calibrated Auto policy beat **already-enabled standalone DFlash K7 and K15**?
3. Are full outputs, continuation state, graph behavior and the 24 GB memory budget still acceptable?

A speedup over true target-only does not answer question 2. A smaller verification buffer does not prove question 1.

## Runtime contract

Only the existing calibrated DFlash2 resident-K15 path gains a new opt-in mode. The model, drafter weights and maximum-capacity persistent buffers remain single/resident. No second model or extra graph family is introduced. Workspace planning now covers every supported target/proposal width because a smaller kernel shape can require different scratch capacity. Consequently this patch does **not** claim reduced allocated VRAM; inspect actual allocator/reservation/device observations.

| Selected action | Legacy/full physical proposal | Selected physical proposal | Target verify width |
| --- | --- | --- | --- |
| K0 | none | none | 1 |
| K7 | 16 columns | 8 columns | 8 |
| K11 | 16 columns | 12 columns | 12 |
| K15 | 16 columns | 16 columns | 16 |

Columns include the anchor. For K7, seven masked columns feed the proposal output head and selector. Attention, dynamic grouped convolution, MLP, head and selector receive the actual reduced shape rather than calculating 16 columns and discarding a prefix. The selected proposal writes directly into the selected target's draft/candidate/probability storage.

`proposal_view.h` binds contiguous lane-major ingress views. Slicing the first eight columns of a resident `[16,B]` tensor would retain the wrong row stride for `B>1`; that is deliberately not the implementation.

The preceding committed-context append remains maximum width. In particular, K15 -> K7 or K15 -> K0 must still append all valid features from the preceding round. Selected target verification, replay and commit continue using the existing per-round `verify_width`. K0 retains the existing target-only-with-resident-DFlash-state path: it skips proposal execution but is **not** the same memory/execution configuration as backend None.

Each finite action already has its own graph family. Compute mode is immutable for an Engine lifetime. The shared ingress/arena are reused sequentially, not concurrently by independent inference traversals. Non-greedy or nonzero-penalty calibrated requests retain the existing K15 fallback, including selected mode. Tree and lookup combinations retain their existing startup rejection. No GDN numerical implementation, acceptance algorithm, sampling tolerance or production default was changed.

## Profile modes and compatibility

Schema 1 is unchanged and means full computation. Schema 2 requires a top-level mode:

```json
{
  "schema_version": 2,
  "artifact_type": "ninfer_spec_router_profile",
  "proposal_compute": "selected",
  "identity": { "...": "exact public startup identity; do not hand-invent this object" },
  "cells": [
    { "active_batch": 1, "frontier_upper": 1024, "draft_tokens": 7 }
  ]
}
```

This is a schema illustration, **not a loadable profile**. The scripts below capture the real identity and write complete controls. Schema 2 also supports `"full"`. Missing/unknown modes, a mode attached to schema 1, unknown schema versions and identity mismatches are rejected.

The calibrated policy is still selected through the existing `--spec-router-profile` interface. Benchmark drivers expose `--proposal-compute full|selected`; default is `full`. No environment variable silently enables selected compute. Switching back to the original schema-1 profile restores the old proposal computation, or keep using standalone Fixed K7 with no calibrated profile.

A selected Auto profile must come from selected-mode measurements. Profile construction validates every raw event's `proposal_width`, `verify_width`, backend, action and neural-skip state. Merely relabeling full K7/K11 measurements as selected does not qualify any cells. The existing uncovered/no-qualified-action K0 behavior is preserved; this is an offline evidence-based table, not a newly invented online acceptance-rate heuristic.

## Local build and regression

No commands in this section were executed as part of the implementation delivery. Use the existing local build configuration and existing model artifact. Build after reconfiguring to register the two new targets:

```bash
export MODEL=/absolute/path/to/model.ninfer
cmake -S . -B build
cmake --build build -j --target \
  ninfer_spec_router_calibration_bench \
  test_speculative_routing_profile \
  test_dflash_proposal_view \
  test_engine_calibrated_routing_real \
  test_engine_calibrated_routing_compact_real

python -m unittest discover -s tests/report_regression -p 'test_*.py'
ctest --test-dir build --output-on-failure \
  -R '^(test_speculative_routing_profile|test_dflash_proposal_view)$'
```

The real-model targets use the project's existing real-model artifact resolution. They can also receive the artifact path explicitly; locate the executable in your configured build tree and pass `"$MODEL"` as its first argument. The compact target compiles the **same** real-model regression source with `NINFER_TEST_COMPACT_DFLASH=1`; it is not a weakened replacement oracle.

Coverage includes K0/7/11/15 against backend None, a within-request frontier switch, exact-B routing during queue drain, cancellation survivor/continuation, small terminal budgets/prefix reuse, non-greedy/penalty K15 fallback, graph-on/off parity and identity rejection. The host proposal-view target checks packed B1/B2/B4/B8 strides and direct output aliases while preserving append storage. Keep running the pre-existing math, rollback and end-to-end output gates as well.

## Experiment A: same-K full vs selected

Set `EXE` to the built native benchmark. The default CMake build places it under `build/bench`; use your actual build output path. NVML device identifiers and CUDA-visible ordinals are distinct on remapped multi-GPU systems.

```bash
export EXE=./build/bench/ninfer_spec_router_calibration_bench
python -m tools.bench.run_dflash_proposal_compute_ab \
  --exe "$EXE" --model "$MODEL" --out out/compute-ab \
  --nvml-device 0 --device 0 \
  --engine-concurrency 8 --concurrency 1,2,4,8 \
  --max-context 32768 --kv-capacity 32768 \
  --actions 0,7,11,15 --pairs 2 --repeats 2 --max-tokens 512 \
  --no-prime-prefix
```

Every action uses the same resident-K15 calibrated startup; only proposal compute mode differs. Pairs alternate full/selected and selected/full with a fresh Engine in every native process. `--prime-prefix` is an alternative explicit campaign, not an unreported warmup. Custom long prompts can be supplied through `--prompt-dir`, with UTF-8 `<workload>.txt` files matching the selected workload names.

`comparison.json` records complete-response equality, actual batch/frontier coverage, paired throughput and p95 ratios. Raw native responses/events/resources, commands, stdout/stderr and sampled device memory remain under each `runs/` directory. K0 and K15 are A/A controls and can never receive `qualified_compute_speedup=true`. A K7/K11 qualification requires complete equality, both execution orders, every pair faster, >=2% paired median improvement, <=5% p95 regression and a common >=95%-dominant actual batch/frontier cell. Report control variance before interpreting small gains.

Requested concurrency does not guarantee the same active GPU batch; actual observations decide coverage. Device-wide `nvidia-smi` samples are observed lower bounds, not exact process peaks. They do not replace Engine logical/allocator/reservation accounting.

## Experiment B: mode-specific calibration, then independent Auto validation

Recalibrate rather than applying historical full-compute cost choices to new selected shapes. Example selected campaign:

```bash
python -m tools.bench.run_resident_spec_router_calibration \
  --exe "$EXE" --model "$MODEL" --out out/selected-calibration \
  --workload-label-prefix selected-short \
  --proposal-compute selected --draft-tokens 7,11,15 \
  --nvml-device 0 --device 0 \
  --engine-concurrency 8 --concurrency 1,2,4,8 \
  --max-context 32768 --kv-capacity 32768 \
  --pairs 2 --repeats 2 --max-tokens 512 --no-prime-prefix

python -m tools.bench.calibrated_router_profile \
  --input out/selected-calibration/measurements.json \
  --output out/selected-profile.json

python -m tools.bench.run_auto_spec_router_comparison \
  --exe "$EXE" --model "$MODEL" --profile out/selected-profile.json \
  --out out/selected-auto-vs-both --compare-both-fixed \
  --nvml-device 0 --device 0 \
  --engine-concurrency 8 --concurrency 1,2,4,8 \
  --max-context 32768 --kv-capacity 32768 \
  --pairs 2 --repeats 2 --max-tokens 512
```

Use the exact same startup identity/capacity when loading the profile. The Auto comparison's old `--fixed-draft-tokens 7` interface remains available, but a single fixed control cannot qualify the new global incremental-value gate.

`--compare-both-fixed` runs fresh true-None / standalone-Fixed / Auto triples against K7 and separately against K15, in both execution orders. It audits physical proposal widths according to profile mode. `incremental_value.qualified_incremental_value` becomes true only when **every requested workload/client group** passes both fixed comparisons and the independently repeated complete outputs agree. A speedup over None alone never sets this flag. Reuse `--proposal-compute full` in calibration and its resulting profile for the old-Auto comparison; a measured new mode is not automatically an improvement.

For long-context coverage, supply independent prompt suites near and across frontier intervals 1024, 8192 and 32768 and the existing targeted transition regressions. Short-prompt coverage is not extrapolated to long prompts; unmeasured cells remain K0. Do not merge mode-mismatched measurement documents using a legacy schema-1-only campaign merger.

## Interpreting completion and failure

Both experiment drivers retain failed runs and return nonzero for execution/structural/output correctness failure. Exit zero means the experiment completed correctly, **not** that the candidate is faster. Inspect `qualified_compute_speedup` and the independent Auto incremental-value gate explicitly. Raw speed ratios may remain as diagnostic evidence when output correctness fails; qualified flags remain false and the supported median is withheld.

Do not promote selected mode if the math/output/continuation gates fail or if the 24 GB memory envelope regresses beyond the deployment budget. If compact fixed-K has a gain but Auto does not beat standalone K7, retain the simpler standalone fixed policy. This change makes those decisions measurable; it does not pre-decide the outcome.
