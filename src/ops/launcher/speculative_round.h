#pragma once

#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void speculative_prepare_verify_inputs_launch(const Tensor& anchors, const Tensor& drafts,
                                              const Tensor& base_positions,
                                              const Tensor& current_extents, Tensor& verify_ids,
                                              Tensor& positions, cudaStream_t stream);
void speculative_prepare_verify_ids_launch(const Tensor& anchors, const Tensor& drafts,
                                           const Tensor& current_extents, Tensor& verify_ids,
                                           cudaStream_t stream);

void speculative_accept_greedy_drafts_launch(const Tensor& target_tokens, const Tensor& logits,
                                             const Tensor& drafts, const Tensor& current_extents,
                                             Tensor& lengths, Tensor& anchors,
                                             Tensor& licensed_tokens, Tensor& licensed_counts,
                                             Tensor& accepted, std::int32_t token_domain,
                                             const SamplingConfig* configs, DeviceSpan workspace,
                                             cudaStream_t stream);

void speculative_accept_sparse_drafts_launch(
    const Tensor& target_tokens, const Tensor& logits, const Tensor& drafts,
    const Tensor& candidate_ids, const Tensor& proposal_q, const Tensor& current_extents,
    Tensor& round_lengths, Tensor& round_anchors, Tensor& licensed_tokens, Tensor& licensed_counts,
    Tensor& accepted_drafts, std::int32_t token_domain, const SamplingConfig* configs,
    bool raw_greedy, DeviceSpan workspace, cudaStream_t stream);

void speculative_select_accepted_hidden_launch(const Tensor& hidden, const Tensor& selectors,
                                               Tensor& out, cudaStream_t stream);
void speculative_tree_gather_bf16_launch(const Tensor& source, const Tensor& path_nodes,
                                         std::int32_t count, Tensor& destination,
                                         cudaStream_t stream);
void speculative_tree_gather_bf16_dynamic_launch(const Tensor& source,
                                                 const Tensor& path_nodes,
                                                 const Tensor& count, Tensor& destination,
                                                 cudaStream_t stream);
void speculative_tree_build_plan_launch(
    const Tensor& candidate_ids, const Tensor& lattice_scores, const Tensor& anchors,
    const Tensor& frontiers, const Tensor& rope_starts, std::int32_t node_budget,
    std::int32_t spine, Tensor& tree_tokens, Tensor& parents, Tensor& depths,
    Tensor& cache_positions, Tensor& rope_positions, cudaStream_t stream);
void speculative_tree_accept_greedy_launch(
    const Tensor& target_tokens, const Tensor& target_logits, const Tensor& tree_tokens,
    const Tensor& parents, std::int32_t live_rows, std::int32_t token_domain,
    Tensor& path_nodes, Tensor& licensed_tokens,
    Tensor& licensed_counts, Tensor& accepted_drafts, Tensor& path_count,
    Tensor& compact_last_index, cudaStream_t stream);
void speculative_make_one_hot_sparse_proposal_launch(
    const Tensor& drafts, const Tensor& current_extents, Tensor& candidate_ids,
    Tensor& proposal_q, std::int32_t token_domain, cudaStream_t stream);

void proposal_remap_token_ids_launch(Tensor& proposal_tokens, const std::int32_t* id_map,
                                     std::int32_t n, cudaStream_t stream);

} // namespace ninfer::ops::detail
