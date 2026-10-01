#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/sampling.h"
#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "core/layout.h"
#include "ops/gdn_ref.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <functional>
#include <queue>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
namespace {
constexpr int S = 128, Hq = 16, W = 8;
const std::vector<std::int32_t> parents{-1, 0, 1, 1, 0, 4, 4, 6};
const std::vector<std::int32_t> accepted_path{0, 4, 6, 7, 0, 0, 0, 0};
const float scale = 1.0F / std::sqrt(128.0F);

std::vector<int> ancestry(int node) {
    std::vector<int> result;
    for (; node >= 0; node = parents[node]) result.push_back(node);
    std::reverse(result.begin(), result.end());
    return result;
}
std::vector<std::uint16_t> pattern(std::size_t n, int seed, float magnitude) {
    std::vector<float> values(n);
    fill_uniform(values, seed, -magnitude, magnitude);
    std::vector<std::uint16_t> result(n);
    for (std::size_t i = 0; i < n; ++i) result[i] = f32_to_bf16(values[i]);
    return result;
}
std::vector<float> decode(const std::vector<std::uint16_t>& bits) {
    std::vector<float> result(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) result[i] = bf16_to_f32(bits[i]);
    return result;
}
std::vector<double> promote(const std::vector<float>& values) {
    return {values.begin(), values.end()};
}
struct Fixture {
    std::vector<std::uint16_t> projected, weight, history, q, k, v;
    std::vector<float> state, g, beta;
};
Fixture fixture(int hv, int kind) {
    Fixture f;
    const int channels = S * (2 * Hq + hv), seed = 6021 + kind * 100;
    f.projected = pattern(channels * W, seed, 0.7F);
    f.weight = pattern(channels * 4, seed + 1, 0.5F);
    f.history = pattern(channels * 3, seed + 2, 0.6F);
    f.state.resize(S * S * hv);
    fill_uniform(f.state, seed + 3, -0.03F, 0.03F);
    f.g.resize(hv * W); f.beta.resize(hv * W);
    fill_uniform(f.g, seed + 4, -0.9F, -0.02F);
    fill_uniform(f.beta, seed + 5, 0.1F, 0.9F);
    return f;
}
gdn_ref::Inputs path_inputs(const Fixture& f, int hv, const std::vector<int>& path) {
    gdn_ref::Inputs in{.head_dim=S, .qk_heads=Hq, .value_heads=hv,
                       .tokens=static_cast<std::int64_t>(path.size())};
    in.state = f.state;
    for (int node : path) {
        for (int d = 0; d < S * Hq; ++d) {
            in.q.push_back(bf16_to_f32(f.q[node * S * Hq + d]));
            in.k.push_back(bf16_to_f32(f.k[node * S * Hq + d]));
        }
        for (int d = 0; d < S * hv; ++d)
            in.v.push_back(bf16_to_f32(f.v[node * S * hv + d]));
        in.g.insert(in.g.end(), f.g.begin() + node * hv, f.g.begin() + (node + 1) * hv);
        in.beta.insert(in.beta.end(), f.beta.begin() + node * hv, f.beta.begin() + (node + 1) * hv);
    }
    return in;
}

int record_case(Fixture& f, int hv, GdnReplayRecordLayer record,
                LinearAttentionStateLayerView states) {
    const int channels = S * (2 * Hq + hv);
    DeviceBuffer projected = to_device(f.projected), weight = to_device(f.weight);
    DeviceBuffer parent = to_device(parents), initial = to_device(std::vector<std::int32_t>{1});
    DeviceBuffer q(S * Hq * W * 2), k(q.bytes), v(S * hv * W * 2), out(v.bytes);
    Tensor tp(projected.p, DType::BF16, {channels, W});
    Tensor tw(weight.p, DType::BF16, {channels, 4});
    Tensor tparent(parent.p, DType::I32, {W}), ti(initial.p, DType::I32, {1});
    Tensor tq(q.p, DType::BF16, {S * Hq, W});
    Tensor tk(k.p, DType::BF16, {S * Hq, W});
    Tensor tv(v.p, DType::BF16, {S * hv, W});
    ops::gdn_projected_tree_conv_record(tp, tw, states.conv, ti, tparent,
                                       record.conv, tq, tk, tv, nullptr);
    cuda_synchronize();
    int failures = 0;
    // Independent depthwise convolution: construct chronological ancestor history, then dot it.
    std::vector<double> expected_q(S * Hq * W), expected_k(expected_q.size()), expected_v(S * hv * W);
    for (int node = 0; node < W; ++node) {
        auto path = ancestry(node);
        for (int channel = 0; channel < channels; ++channel) {
            std::vector<double> sequence;
            for (int t = 0; t < 3; ++t) sequence.push_back(bf16_to_f32(f.history[t * channels + channel]));
            for (int ancestor : path) sequence.push_back(bf16_to_f32(f.projected[ancestor * channels + channel]));
            double sum = 0;
            for (int t = 0; t < 4; ++t)
                sum += sequence[sequence.size() - 4 + t] * bf16_to_f32(f.weight[t * channels + channel]);
            const double silu = sum / (1.0 + std::exp(-sum));
            if (channel < S * Hq) expected_q[node * S * Hq + channel] = silu;
            else if (channel < 2 * S * Hq) expected_k[node * S * Hq + channel - S * Hq] = silu;
            else expected_v[node * S * hv + channel - 2 * S * Hq] = silu;
        }
    }
    const ReductionCriterion bf16{4.1e-3, 5e-6, kBf16GrossRelativeFloor};
    failures += verify_reduction("tree conv q", from_device_bf16(q, expected_q.size()), expected_q, bf16);
    failures += verify_reduction("tree conv k", from_device_bf16(k, expected_k.size()), expected_k, bf16);
    failures += verify_reduction("tree conv v", from_device_bf16(v, expected_v.size()), expected_v, bf16);
    failures += verify_exact("tree raw conv", from_device<std::uint16_t>(record.conv.data, f.projected.size()), f.projected);
    // The public BF16 convolution outputs are operands for the independent recurrence oracle.
    f.q = from_device<std::uint16_t>(q, S * Hq * W);
    f.k = from_device<std::uint16_t>(k, S * Hq * W);
    f.v = from_device<std::uint16_t>(v, S * hv * W);
    DeviceBuffer dg = to_device(f.g), db = to_device(f.beta);
    tq = tq.view({S, Hq, W, 1}); tk = tk.view({S, Hq, W, 1}); tv = tv.view({S, hv, W, 1});
    Tensor tg(dg.p, DType::FP32, {hv, W, 1}), tb(db.p, DType::FP32, {hv, W, 1});
    Tensor tout(out.p, DType::BF16, {S, hv, W, 1});
    ops::gated_delta_net_tree_replay_record(tq, tk, tv, tg, tb, scale, states.recurrent,
                                           ti, tparent, record.key, record.value, record.gate, tout, nullptr);
    cuda_synchronize();
    std::vector<double> expected_out(S * hv * W);
    for (int node = 0; node < W; ++node) {
        const auto path = ancestry(node);
        const auto oracle = gdn_ref::evaluate(path_inputs(f, hv, path), scale, true);
        std::copy(oracle.out.end() - S * hv, oracle.out.end(), expected_out.begin() + node * S * hv);
    }
    failures += verify_reduction("tree GDN output", from_device_bf16(out, expected_out.size()), expected_out, bf16);
    failures += verify_exact("tree raw key", from_device<std::uint16_t>(record.key.data, f.k.size()), f.k);
    failures += verify_exact("tree raw value", from_device<std::uint16_t>(record.value.data, f.v.size()), f.v);
    std::vector<std::uint32_t> gates(2 * f.g.size());
    for (std::size_t i = 0; i < f.g.size(); ++i) {
        gates[2 * i] = std::bit_cast<std::uint32_t>(f.g[i]);
        gates[2 * i + 1] = std::bit_cast<std::uint32_t>(f.beta[i]);
    }
    failures += verify_exact("tree raw gates", from_device<std::uint32_t>(record.gate.data, gates.size()), gates);
    return failures;
}

int run_geometry(int layers, int hv) {
    const int channels = S * (2 * Hq + hv);
    LayoutBuilder rb;
    const auto rl = plan_gdn_replay_records(rb, {.layers=layers, .record_capacity=1,
        .width=W, .conv_channels=channels, .qk_heads=Hq, .value_heads=hv, .key_dim=S, .value_dim=S});
    DeviceBuffer rs(rb.finish(256)); rs.fill(0xff);
    GdnReplayRecords records({rs.p, rs.bytes}, rl);
    LayoutBuilder sb;
    const auto sl = plan_linear_attention_state_pool(sb, {.layers=static_cast<std::uint32_t>(layers),
        .conv_channels=channels, .conv_width=3, .value_heads=hv, .value_head_dim=S,
        .key_head_dim=S, .slot_count=3, .conv_dtype=DType::BF16});
    DeviceBuffer ss(sb.finish(256)); ss.fill(0);
    LinearAttentionStatePool states({ss.p, ss.bytes}, sl);
    std::vector<Fixture> fixtures;
    for (int kind = 0; kind < 3; ++kind) fixtures.push_back(fixture(hv, kind));
    const auto kind_of = [layers](int layer) { return layer == 0 ? 0 : layer == layers - 1 ? 2 : 1; };
    for (int layer = 0; layer < layers; ++layer) {
        const auto& f = fixtures[kind_of(layer)];
        auto recurrent = states.recurrent_slot(layer, 1), conv = states.conv_slot(layer, 1);
        cuda_check(cudaMemcpy(recurrent.data, f.state.data(), recurrent.bytes(), cudaMemcpyHostToDevice), "upload initial state");
        cuda_check(cudaMemcpy(conv.data, f.history.data(), conv.bytes(), cudaMemcpyHostToDevice), "upload initial history");
    }
    int failures = 0;
    for (int layer : {0, 1, layers - 1})
        failures += record_case(fixtures[kind_of(layer)], hv, records.layer(layer, 1), states.layer_view(layer));
    for (int layer = 2; layer < layers - 1; ++layer) {
        const auto src = records.layer(1, 1), dst = records.layer(layer, 1);
        for (auto pair : {std::pair{src.conv, dst.conv}, std::pair{src.key, dst.key},
                          std::pair{src.value, dst.value}, std::pair{src.gate, dst.gate}})
            cuda_check(cudaMemcpy(pair.second.data, pair.first.data, pair.first.bytes(), cudaMemcpyDeviceToDevice), "replicate layer records");
    }
    const ops::GdnReplayFoldPlan plan(records, states.all_layers_view());
    DeviceBuffer path = to_device(accepted_path);
    Tensor tpath(path.p, DType::I32, {W});
    const auto records_before = from_device<std::uint8_t>(rs, rs.bytes);
    const ReductionCriterion state_criterion{2e-6, 2e-7, 2e-6};
    for (int count : {0, 1, 2, 3, 4}) {
        states.zero_slot(2);
        plan.execute_tree(1, 2, count == 0 ? Tensor{} : tpath, count, nullptr);
        cuda_synchronize();
        for (int kind = 0; kind < 3; ++kind) {
            const auto& f = fixtures[kind];
            std::vector<double> expected_state(f.state.size(), 0.0);
            std::vector<std::uint16_t> expected_history(channels * 3, 0);
            if (count > 0) {
                const std::vector<int> prefix(accepted_path.begin(), accepted_path.begin() + count);
                expected_state = gdn_ref::evaluate(path_inputs(f, hv, prefix), scale, true).final_state;
                std::vector<std::uint16_t> sequence = f.history;
                for (int node : prefix)
                    sequence.insert(sequence.end(), f.projected.begin() + node * channels,
                                    f.projected.begin() + (node + 1) * channels);
                std::copy(sequence.end() - channels * 3, sequence.end(), expected_history.begin());
            }
            for (int layer = 0; layer < layers; ++layer) {
                if (kind_of(layer) != kind) continue;
                const auto dest = states.recurrent_slot(layer, 2), conv = states.conv_slot(layer, 2);
                const auto actual = from_device<float>(dest.data, f.state.size());
                const auto label = "tree fold Hv=" + std::to_string(hv) + " layer=" + std::to_string(layer) + " count=" + std::to_string(count);
                failures += verify_reduction(label, promote(actual), expected_state, state_criterion);
                failures += verify_exact((label + " history").c_str(), from_device<std::uint16_t>(conv.data, expected_history.size()), expected_history);
                const auto source = states.recurrent_slot(layer, 1), source_conv = states.conv_slot(layer, 1);
                failures += verify_exact((label + " source state").c_str(), from_device<float>(source.data, f.state.size()), f.state);
                failures += verify_exact((label + " source history").c_str(), from_device<std::uint16_t>(source_conv.data, f.history.size()), f.history);
                const auto unused = states.recurrent_slot(layer, 0), unused_conv = states.conv_slot(layer, 0);
                failures += verify_exact((label + " inactive state").c_str(), from_device<float>(unused.data, f.state.size()), std::vector<float>(f.state.size(), 0));
                failures += verify_exact((label + " inactive history").c_str(), from_device<std::uint16_t>(unused_conv.data, f.history.size()), std::vector<std::uint16_t>(f.history.size(), 0));
            }
        }
    }
    failures += verify_exact("tree fold records read only", from_device<std::uint8_t>(rs, rs.bytes), records_before);
    return failures;
}

struct TreeCandidate {
    std::vector<int> ranks;
    double probability;
};

int run_build_plan() {
    constexpr int physical = 16, candidates = 16;
    constexpr std::int32_t anchor = 317, frontier = 29, rope_start = 83;
    int failures = 0;
    for (int active : {3, 7, 11}) {
        std::vector<std::int32_t> ids(candidates * active);
        std::vector<float> lattice(candidates * candidates * active);
        for (int step = 0; step < active; ++step) {
            for (int rank = 0; rank < candidates; ++rank)
                ids[step * candidates + rank] = 1000 + step * 32 + rank;
            for (int predecessor = 0; predecessor < candidates; ++predecessor) {
                const int best = (step * 3 + predecessor * 5 + 2) % candidates;
                for (int rank = 0; rank < candidates; ++rank) {
                    const int distance = (rank - best + candidates) % candidates;
                    lattice[(step * candidates + predecessor) * candidates + rank] =
                        -0.75F * distance - 0.03F * ((predecessor * 7 + step * 11 + rank * 13) % 17);
                }
            }
        }
        // Independent FP64 conditional probabilities from the represented FP32 scores.
        std::vector<double> probabilities(lattice.size());
        for (std::size_t base = 0; base < lattice.size(); base += candidates) {
            double sum = 0;
            for (int rank = 0; rank < candidates; ++rank) {
                probabilities[base + rank] = std::exp(static_cast<double>(lattice[base + rank]));
                sum += probabilities[base + rank];
            }
            for (int rank = 0; rank < candidates; ++rank) probabilities[base + rank] /= sum;
        }
        const auto children = [&](const TreeCandidate& node) {
            std::vector<TreeCandidate> result;
            const int step = static_cast<int>(node.ranks.size());
            if (step == active) return result;
            const int predecessor = node.ranks.empty() ? 0 : node.ranks.back();
            for (int rank = 0; rank < candidates; ++rank) {
                auto path = node.ranks; path.push_back(rank);
                result.push_back({std::move(path), node.probability *
                    probabilities[(step * candidates + predecessor) * candidates + rank]});
            }
            return result;
        };
        for (int spine : {1, std::min(3, active)}) {
            // Force the greedy spine, then select additional complete paths best-first.
            std::vector<TreeCandidate> selected{{{}, 1.0}};
            const auto less = [](const TreeCandidate& a, const TreeCandidate& b) {
                return a.probability < b.probability;
            };
            std::priority_queue<TreeCandidate, std::vector<TreeCandidate>, decltype(less)> frontier_nodes(less);
            TreeCandidate current{{}, 1.0};
            for (int depth = 0; depth < spine; ++depth) {
                const auto options = children(current);
                const auto best = std::max_element(options.begin(), options.end(), less);
                for (const auto& option : options)
                    if (option.ranks != best->ranks) frontier_nodes.push(option);
                current = *best;
                selected.push_back(current);
            }
            while (selected.size() < static_cast<std::size_t>(active + 1)) {
                auto next = frontier_nodes.top(); frontier_nodes.pop();
                selected.push_back(next);
                for (const auto& option : children(next)) frontier_nodes.push(option);
            }
            std::vector<std::int32_t> expected_tokens(physical, anchor), expected_parents(physical, -1);
            std::vector<std::int32_t> expected_depths(physical, 0), expected_cache(physical, frontier), expected_rope(physical, rope_start);
            int output_index = 0;
            // DFS in descending path-probability order, independently remapping logical paths.
            std::function<void(const TreeCandidate&, int)> emit = [&](const TreeCandidate& node, int parent) {
                const int index = output_index++;
                const int depth = static_cast<int>(node.ranks.size());
                expected_parents[index] = parent;
                expected_depths[index] = depth;
                expected_cache[index] += depth;
                expected_rope[index] += depth;
                if (depth) expected_tokens[index] = ids[(depth - 1) * candidates + node.ranks.back()];
                std::vector<TreeCandidate> selected_children;
                for (const auto& candidate : selected) {
                    if (candidate.ranks.size() == node.ranks.size() + 1 &&
                        std::equal(node.ranks.begin(), node.ranks.end(), candidate.ranks.begin()))
                        selected_children.push_back(candidate);
                }
                std::sort(selected_children.begin(), selected_children.end(),
                    [](const auto& a, const auto& b) { return a.probability > b.probability; });
                for (const auto& child : selected_children) emit(child, index);
            };
            emit(selected.front(), -1);
            DeviceBuffer di = to_device(ids), dl = to_device(lattice);
            DeviceBuffer da = to_device(std::vector<std::int32_t>{anchor});
            DeviceBuffer df = to_device(std::vector<std::int32_t>{frontier});
            DeviceBuffer dr = to_device(std::vector<std::int32_t>{rope_start});
            DeviceBuffer tokens(physical * 4), parent(physical * 4), depth(physical * 4), cache(physical * 4), rope(physical * 4);
            for (auto* output : {&tokens, &parent, &depth, &cache, &rope}) output->fill(0xa5);
            Tensor ti(di.p, DType::I32, {candidates, active});
            Tensor tl(dl.p, DType::FP32, {candidates, candidates, active});
            Tensor ta(da.p, DType::I32, {1}), tf(df.p, DType::I32, {1}), tr(dr.p, DType::I32, {1});
            Tensor tt(tokens.p, DType::I32, {physical}), tp(parent.p, DType::I32, {physical}), td(depth.p, DType::I32, {physical});
            Tensor tc(cache.p, DType::I32, {physical}), tro(rope.p, DType::I32, {physical});
            ops::speculative_tree_build_plan(ti, tl, ta, tf, tr, active, spine, tt, tp, td, tc, tro, nullptr);
            cuda_synchronize();
            const auto label = "tree build active=" + std::to_string(active) + " physical=16 spine=" + std::to_string(spine);
            failures += verify_exact((label + " tokens").c_str(), from_device<std::int32_t>(tokens, physical), expected_tokens);
            failures += verify_exact((label + " parents").c_str(), from_device<std::int32_t>(parent, physical), expected_parents);
            failures += verify_exact((label + " depths").c_str(), from_device<std::int32_t>(depth, physical), expected_depths);
            failures += verify_exact((label + " cache").c_str(), from_device<std::int32_t>(cache, physical), expected_cache);
            failures += verify_exact((label + " rope").c_str(), from_device<std::int32_t>(rope, physical), expected_rope);
        }
    }
    return failures;
}

int run_accept_gather() {
    const std::vector<std::int32_t> tokens{10, 11, 12, 13, 14, 15, 16, 17};
    int failures = 0;
    constexpr int rows = 129, dest_width = W + 2;
    for (auto targets : {std::vector<std::int32_t>{14, 0, 0, 0, 16, 0, 17, 20},
                         std::vector<std::int32_t>{14, 0, 0, 0, 21, 0, 17, 20},
                         std::vector<std::int32_t>{22, 0, 0, 0, 16, 0, 17, 20}}) {
        std::vector<std::int32_t> expected_path(W, 0), expected_tokens(W, 0);
        int current = 0, accepted = 0, count = 0;
        while (true) {
            expected_path[count] = current;
            const int wanted = targets[current];
            expected_tokens[count++] = wanted;
            const auto child = std::find_if(tokens.begin(), tokens.end(), [&](const auto& token) {
                const auto index = static_cast<int>(&token - tokens.data());
                return parents[index] == current && token == wanted;
            });
            if (child == tokens.end()) break;
            current = static_cast<int>(child - tokens.begin());
            ++accepted;
        }
        DeviceBuffer dt = to_device(targets), tree = to_device(tokens), dp = to_device(parents);
        DeviceBuffer logits = to_device(pattern(32 * W, 7881, 0.3F));
        DeviceBuffer path(dest_width * 4), licensed(W * 4), lc(4), ad(4), pc(4), last(4);
        path.fill(0xff); licensed.fill(0xff);
        Tensor tt(dt.p, DType::I32, {W}), tr(tree.p, DType::I32, {W}), parent(dp.p, DType::I32, {W});
        Tensor tl(logits.p, DType::BF16, {32, W});
        Tensor tp(path.p, DType::I32, {W}), tok(licensed.p, DType::I32, {W});
        Tensor tc(lc.p, DType::I32, {1}), ta(ad.p, DType::I32, {1}), tpc(pc.p, DType::I32, {1}), tlast(last.p, DType::I32, {1});
        ops::speculative_tree_accept_greedy(tt, tl, tr, parent, W, 32, tp, tok, tc, ta, tpc, tlast, nullptr);
        cuda_synchronize();
        failures += verify_exact("tree accept path", from_device<std::int32_t>(path, W), expected_path);
        failures += verify_exact("tree accept tokens", from_device<std::int32_t>(licensed, W), expected_tokens);
        for (auto pair : {std::pair{lc.p, count}, std::pair{ad.p, accepted}, std::pair{pc.p, count}, std::pair{last.p, count - 1}})
            failures += verify_exact("tree accept scalar", from_device<std::int32_t>(pair.first, 1), std::vector<std::int32_t>{pair.second});
        Tensor gather_path(path.p, DType::I32, {dest_width});
        const auto bits = pattern(rows * W, 7781, 0.2F);
        DeviceBuffer src = to_device(bits), dst(rows * dest_width * 2);
        Tensor source(src.p, DType::BF16, {rows, W}), dest(dst.p, DType::BF16, {rows, dest_width});
        // Frontend terminal truncation lowers the device count before gather/commit.
        for (int partial = 0; partial <= count; ++partial) {
            pc.copy_from_host(&partial, sizeof(partial)); dst.fill(0xff);
            ops::speculative_tree_gather_bf16_dynamic(source, gather_path, tpc, dest, nullptr);
            std::vector<std::uint16_t> expected(rows * dest_width, 0);
            for (int i = 0; i < partial; ++i)
                std::copy_n(bits.begin() + expected_path[i] * rows, rows, expected.begin() + i * rows);
            failures += verify_exact("tree gather accepted prefix", from_device<std::uint16_t>(dst, expected.size()), expected);
        }
    }
    // The public finite/domain gate must publish the failure sentinel instead of a draft.
    for (int wanted : {14, 32, -1}) {
        std::vector<std::int32_t> targets(W, 0); targets[0] = wanted;
        auto logit_bits = pattern(32 * W, 7891, 0.3F);
        if (wanted == 14) logit_bits[wanted] = 0x7fc0; // represented BF16 NaN
        DeviceBuffer dt = to_device(targets), tree = to_device(tokens), dp = to_device(parents);
        DeviceBuffer logits = to_device(logit_bits);
        DeviceBuffer path(W * 4), licensed(W * 4), lc(4), ad(4), pc(4), last(4);
        path.fill(0xff); licensed.fill(0xff);
        Tensor tt(dt.p, DType::I32, {W}), tr(tree.p, DType::I32, {W}), parent(dp.p, DType::I32, {W});
        Tensor tl(logits.p, DType::BF16, {32, W});
        Tensor tp(path.p, DType::I32, {W}), tok(licensed.p, DType::I32, {W});
        Tensor tc(lc.p, DType::I32, {1}), ta(ad.p, DType::I32, {1}), tpc(pc.p, DType::I32, {1}), tlast(last.p, DType::I32, {1});
        ops::speculative_tree_accept_greedy(tt, tl, tr, parent, W, 32, tp, tok, tc, ta, tpc, tlast, nullptr);
        cuda_synchronize();
        std::vector<std::int32_t> expected_tokens(W, 0);
        expected_tokens[0] = ops::kSamplerNonFiniteToken;
        failures += verify_exact("tree invalid target sentinel", from_device<std::int32_t>(licensed, W), expected_tokens);
        failures += verify_exact("tree invalid target path", from_device<std::int32_t>(path, W), std::vector<std::int32_t>(W, 0));
        for (auto pair : {std::pair{lc.p, 1}, std::pair{ad.p, 0}, std::pair{pc.p, 1}, std::pair{last.p, 0}})
            failures += verify_exact("tree invalid target scalar", from_device<std::int32_t>(pair.first, 1), std::vector<std::int32_t>{pair.second});
    }
    return failures;
}
} // namespace
int main() {
    if (cuda_unavailable()) return 77;
    int failures = run_build_plan();
    failures += run_accept_gather();
    failures += run_geometry(48, 48);
    failures += run_geometry(30, 32);
    std::cout << (failures ? "FAIL" : "OK") << " tree_branch_commit\n";
    return failures ? 1 : 0;
}
