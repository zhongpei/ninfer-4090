# Third-party notices

## TandemLLM

The adaptive speculative-routing work in this repository incorporates and adapts design and source
ideas from [0xBakeer/TandemLLM](https://github.com/0xBakeer/TandemLLM), reviewed on 2026-09-30 at
commit `c11aecaa7ba767642409ff51d90d48c79c572d3a`.

TandemLLM engine source is licensed under **AGPL-3.0-only** (or a separate commercial license from
its copyright holder). This repository is licensed as AGPL-3.0-only so adapted source can remain
under compatible terms.

TandemLLM copyright notices and attribution remain with their respective authors. In particular,
StairCut is described by Khaled Bakeer in *StairCut: sizing speculative draft trees on a measured
verification-cost staircase* (2026).

The NInfer adaptations also use TandemLLM's counted n-gram/suffix-memory design ideas: local
occurrence voting, a larger static suffix corpus, a persistent cross-request suffix store,
confidence-gated copy drafting, neural-drafter replacement/head-skip experiments, and deeper copy
chains after strong acceptance. NInfer reimplements these around its existing Program transaction,
ReplaySSM and paged-KV contracts rather than importing Tandem's Python engine wholesale.

The DFlash2 training tools also adapt TandemLLM's target-tap recorder contract, CE + target-top-K
KL objective, block-16 fine-tuning, held-out accepted-tokens-per-block gate, resume/export workflow,
and DFlash2 lattice/tree-planning algorithms. The standalone NInfer copy removes the Tandem engine
dependency and exports the companion tensor names already understood by NInfer's converter.

Runtime tree verification additionally adapts TandemLLM's speculative-tree and recurrent-tree
verification ideas. NInfer's implementation is specific to its hybrid target: DFS depth-position
reuse for temporary full-attention KV, ancestor-aware GDN convolution/recurrent evaluation, and
accepted-path-only ReplaySSM/StateImage commit.

The final 24GB integration also adapts TandemLLM's server-lifetime router-learning and
deterministic sampled-tree concepts. NInfer's sampled tree uses its existing position-keyed
counter RNG and deliberately falls back to chain verification when branch-specific penalty state
would be required.

This notice covers code/design adapted from TandemLLM. Model weights and other third-party
components retain their own licenses and notices.
