# Runtime DFlash2 tree verification

NInfer can verify a small DFlash2 candidate tree in one target traversal instead of collapsing the
selector lattice to a single chain. The first runtime version is deliberately narrow so the hybrid
target state remains exact:

- DFlash2 only;
- one active request (C1);
- raw greedy target sampling (temperature <= 0, no presence/frequency penalty);
- at most 15 draft nodes / 16 target rows;
- node budget equals startup `--draft-tokens`;
- opt-in; all unsupported rounds fall back to the existing chain path.

Enable it with:

```bash
ninfer-serve MODEL.ninfer \
  --spec dflash2 --draft-tokens 15 \
  --spec-tree lattice --spec-tree-nodes 15 --spec-tree-spine 7
```

The default remains `--spec-tree off`.

## Proposal tree

DFlash2 already computes a 16-candidate conditional lattice for the candidate selector. Tree mode
retains those edge scores instead of discarding them after choosing one chain.

The device planner (one tiny control CTA on the decode stream):

1. installs a greedy lattice spine;
2. scores alternative edges by cumulative path probability;
3. spends the remaining node budget best-first;
4. reorders the selected nodes into DFS preorder.

The root is the already-known target anchor. A 15-node tree therefore consumes 16 target input rows,
the same physical target width as chain K15.

`--spec-tree-spine` controls how many greedy levels are installed before alternatives compete for
the remaining budget.

## Full attention without per-node StateImage copies

The target's full-attention layers reuse temporary cache positions by tree depth.

For a DFS tree such as:

```text
root
  A
    B
    C
  D
```

temporary positions are:

```text
root = frontier
A,D  = frontier + 1
B,C  = frontier + 2
```

Each node is evaluated in DFS order. Before a sibling is evaluated it overwrites only its own depth
and deeper temporary cache rows; all ancestors remain intact. Since causal attention only reads keys
at or before the node's absolute position, stale deeper sibling rows are invisible.

Every node's normalized K and V are also retained in a small BF16 replay plane. After target
acceptance the chosen path is gathered and republished into canonical contiguous positions:

```text
frontier, frontier+1, ..., frontier+committed-1
```

Only that canonical prefix becomes Program-owned KV state.

This avoids allocating one complete paged-KV/StateImage replica per tree node.

## GDN / ReplaySSM

Qwen3.8 is hybrid, so tree verification cannot be implemented as an attention mask alone.

Tree mode adds two ancestor-aware GDN operations.

### Dynamic convolution

For every tree node the 4-tap convolution reads:

```text
tail_3(round-entry history + node ancestors) + current projected input
```

It never treats the previous DFS storage row as the previous logical token.

### Recurrent state

Each node output is evaluated from the immutable round-entry recurrent state plus exactly the
root-to-node transition path. Raw K/V/gate/conv transition records remain indexed by tree node.

When the Frontend commits a generated prefix, `GdnReplayFoldPlan::execute_tree` consumes
`path_nodes[0:commit_columns]` and publishes only that path into the destination StateImage.
A terminal partial-prefix commit simply lowers `commit_columns`.

The first implementation recomputes the short ancestor path per node rather than materializing a
full recurrent state per node. With the runtime cap of 16 rows this trades a small bounded compute
cost for avoiding multi-GB state replication.

## Acceptance and commit contract

For tree verification:

```text
licensed_count = accepted_drafts + 1
```

The accepted target input state is exactly:

```text
root + accepted draft nodes
```

which is exactly `licensed_count` tree nodes.

The same `path_nodes` prefix is therefore used for:

- ReplaySSM/GDN state fold;
- target full-attention KV compaction;
- DFlash target-tap compaction;
- continuation hidden selection.

The correction/bonus token itself remains the next unexecuted ledger token, matching the existing
chain speculative contract.

## DFlash target features

The five target taps needed by the next DFlash2 round are initially captured in DFS-node order.
After acceptance they are gathered into accepted-path order and written back to the existing
`pending_features` buffer. No second DFlash context format is introduced.

Target hidden rows are compacted the same way. This keeps the existing Frontend partial-terminal
hidden correction contract (`selector = committed - 1`) unchanged.

## Fallback behavior

Requesting tree mode does not force unsupported requests onto an approximate path.

Tree v1 falls back to chain verification when any of these is true:

- decode batch has more than one row;
- lookup replace/head-skip owns the proposal;
- adaptive/budget/context truncation makes this round narrower than startup K;
- positive temperature is used;
- presence or frequency penalties are active.

The request metrics expose `tree_rounds` and `tree_fallback_rounds` separately.

Tree mode currently bypasses the neural target CUDA Graph because lattice-to-tree construction is
data-dependent. Tree construction and raw-greedy acceptance stay on device, so the runtime no longer
performs the two mid-round D2H synchronizations used by the correctness-first implementation.
Chain mode remains graphed.

## A/B profiles

Use identical model, prompt, KV format and greedy settings.

```text
A: --spec-tree off
B: --draft-tokens 7  --spec-tree lattice --spec-tree-nodes 7  --spec-tree-spine 5
C: --draft-tokens 11 --spec-tree lattice --spec-tree-nodes 11 --spec-tree-spine 7
D: --draft-tokens 15 --spec-tree lattice --spec-tree-nodes 15 --spec-tree-spine 7
```

The runtime reports:

- tree rounds;
- chain fallback rounds;
- tree nodes proposed;
- tree accepted drafts;
- ordinary DFlash2 accepted tokens / round;
- decode throughput.

The supplied `scripts/sweeps/dflash2-tree-realtext.ps1` runs the fixed-chain and tree arms on the
same production-shaped prompt and records output hashes.

## Interaction with Stair and lookup

Tree v1 is a separate A/B arm.

- Strong lookup takeover continues to use the copy chain; it does not build a DFlash tree.
- With `--spec-router fixed`, tree mode uses the configured `--spec-tree-nodes` budget.
- With `--spec-router stair`, the same measured Stair controller becomes **Tree-StairCut**:
  the physical DFlash2/round state remains K15/16 rows, while target-tree verification snaps to
  the largest admissible configured tier (default 3/7/11/15 nodes).
- Tree acceptance feeds the Stair observation, so the next tree round can move between tiers.
- When a requested tree round cannot run, the existing chain fallback remains available.

This is specifically intended for 24 GB cards: it changes verify compute, not resident model state,
and does not allocate a second drafter or a 24/32-row tree buffer.

## Provenance

The tree proposal and recurrent-state approach adapts the DFlash2 lattice/tree and DFS recurrent
verification ideas from 0xBakeer/TandemLLM. NInfer reimplements the state transaction around its
Paged KV, ReplaySSM, StateImage, Frontend commit, and DFlash context contracts.
