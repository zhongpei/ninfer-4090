# TandemLLM integration status for 24GB RTX 4090

This document closes the TandemLLM absorption series for the 24GB target profile. It distinguishes
features that are intentionally excluded from features that are still required for the selected
architecture.

## Selected production shape

The 24GB profile is:

```text
one resident DFlash2-b16 drafter
        +
maximum 16 target tree rows
        +
Tree-Stair 3/7/11/15 active nodes
        +
lookup vote/head-skip/deep copy
        +
paged target KV + ReplaySSM
```

No second neural drafter and no 24/32-row persistent tree state are reserved.

A representative command after calibrating the machine is:

```bash
ninfer-serve MODEL.ninfer \
  --spec dflash2 --draft-tokens 15 \
  --spec-tree lattice --spec-tree-nodes 15 --spec-tree-spine 7 \
  --spec-router stair \
  --spec-stair-widths 3,7,11,15 \
  --spec-stair-costs C3,C7,C11,C15 \
  --lookup-ngram 5 --lookup-strategy vote --lookup-dflash skip \
  --lookup-persistent-tokens 262144 \
  --lookup-persistent-path /var/lib/ninfer/lookup.tokens
```

## Completed

### Licensing and A/B controls

- repository-wide AGPL-3.0-only
- TandemLLM attribution
- every imported speculative policy is opt-in and has a fixed/off baseline

### DFlash2

- K=1..15 runtime
- block-16 training path
- teacher taps 5/19/33/47/61
- CE + target-top-K KL objective
- partial/full train modes, resume and export
- existing NInfer companion conversion
- `ninfer_gate.py` evaluates the final converted artifact through the real NInfer runtime

### Chain and tree routing

- fixed chain baseline
- chain Stair
- runtime DFlash2 lattice tree
- hybrid target tree state through full attention and ReplaySSM/GDN
- GPU lattice-to-tree build
- GPU tree acceptance/path walk
- device-count hidden/tap/KV accepted-path compaction
- Tree-Stair target budgets 3/7/11/15 while the single b16 drafter stays resident
- request-local plus server-lifetime routing evidence

### Sampling

- greedy runtime tree
- positive-temperature deterministic tree when presence/frequency penalties are zero
- target draws use NInfer's position-keyed counter RNG and the ordinary decode purpose
- penalty requests fall back to the existing chain path rather than approximate branch counts

### Lookup

- historical recent-match lookup
- counted multi-source vote
- process history
- optional restart-persistent process history
- static suffix corpus
- DFlash replace/head-skip
- deep copy after strong acceptance

### Calibration

- real-text chain/tree sweeps
- one-resident-b16 fixed tree3/7/11/15 measurements
- `tools/calibrate_spec_4090.py` derives the local Tree-Stair cost table

## Deliberately excluded from the 24GB profile

### A second resident b8 drafter

Tandem can load narrow and wide drafters. NInfer keeps only b16 resident and cuts target work.
Keeping b8 duplicates neural-drafter weights for a small routing benefit and is the wrong trade on
a 24GB card.

### 24/32-row persistent tree buffers

Tandem verifies up to 32 rows. NInfer's selected profile stops at 16 physical rows. Wider trees
increase target tree K/V replay, ReplaySSM records, workspace/graph families and validation surface.
They can be revisited on a larger-memory GPU, but are not part of the 4090 baseline.

### Blackwell NVFP4/TMA kernels

Tandem's GB10/SM121-specific NVFP4/TMA paths are not RTX 4090 targets. NInfer keeps its Ada T2/Q5/Q8
weight routes.

### Lookup-tree + DFlash-tree merge

The current 15-node target budget is deliberately scarce. Strong copy runs already bypass the
neural drafter through head-skip; weak copy evidence yields to the DFlash tree. A merged tree would
spend the same 15 nodes on both sources and adds substantial policy/merge complexity. This remains
an optional experiment rather than part of the selected 24GB production route.

## Remaining performance experiments, not correctness gaps

The runtime tree path is exact for its supported contract, but two kernel-level experiments may
still improve speed without changing the architecture:

1. a fused multi-node tree full-attention kernel instead of invoking the existing single-node
   causal path per DFS node;
2. a DFS depth-stack GDN tree kernel that carries ancestor state once instead of replaying the
   short root-to-node path for every node.

These should be justified by the RTX 4090 tree sweep after the GPU control-path changes. They do
not require a different Program/StateImage contract and therefore are optimization work, not
missing functionality.

## Training caveat

The differentiable teacher recorder currently reads Hugging Face target taps/distributions.
A T2/Q5/Ternary conversion may move near-tie logits. The final converted companion must therefore
pass `tools/dflash2_training/ninfer_gate.py` against the actual `.ninfer` artifact.

A future native NInfer tap exporter could distill directly from quantized runtime taps, but that
would add a diagnostic model-export surface to the serving engine. It is not required to train and
ship a custom DFlash2 checkpoint under the current workflow.

## Recommended closure workflow

1. train b16 from recorded target data;
2. convert it into the intended T2/Q5/Ternary target artifact;
3. gate released-vs-custom artifacts with `ninfer_gate.py`;
4. run `dflash2-tree-realtext.ps1` on the target RTX 4090 and selected KV format;
5. derive 3/7/11/15 costs with `calibrate_spec_4090.py`;
6. serve the single-b16 Tree-Stair profile;
7. retain lookup/router history across requests and optionally across restarts.

At that point further Tandem work is optional tuning rather than a missing part of the 24GB design.
