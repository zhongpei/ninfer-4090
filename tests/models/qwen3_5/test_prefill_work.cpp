#include "models/qwen3_5/program/planning/prefill_work.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

namespace {
struct Capture { std::uint32_t frontier; };
using ninfer::models::qwen3_5::runtime_support::segmented_prefill_units;

// Execute the cursor/remaining contract: captures return to the scheduler, whereas
// rewrite boundaries only split the remaining nominal chunk. Enumerate both
// scheduler budgets in auto mode; no model or scheduler results are mocked.
std::uint64_t execution_units(std::uint32_t cursor, std::uint32_t end,
                             std::uint32_t chunk, std::uint32_t quantum,
                             const std::vector<Capture>& captures,
                             const std::vector<std::uint32_t>& rewrites, bool automatic) {
    if (cursor == end) { return 0; }
    std::uint64_t maximum = 0;
    for (const auto budget : {chunk, automatic ? quantum : chunk}) {
        auto position = cursor;
        auto remaining = std::min(budget, end - position);
        std::uint64_t units = 0;
        while (remaining != 0) {
            auto stop = position + remaining;
            for (const auto capture : captures) {
                if (capture.frontier > position) { stop = std::min(stop, capture.frontier); }
            }
            for (const auto rewrite : rewrites) {
                if (rewrite > position) { stop = std::min(stop, rewrite); }
            }
            // Independently count service quanta in the processed token interval.
            for (auto token = position; token < stop; token += std::min(quantum, stop-token)) {
                ++units;
            }
            remaining -= stop - position;
            position = stop;
            if (std::any_of(captures.begin(), captures.end(), [position](Capture capture) {
                    return capture.frontier == position;
                })) { break; }
        }
        maximum = std::max(maximum, units + execution_units(position, end, chunk, quantum,
                                                             captures, rewrites, automatic));
        if (!automatic) { break; }
    }
    return maximum;
}

bool check(std::uint32_t begin, std::uint32_t end, std::uint32_t chunk, std::uint32_t quantum,
           std::vector<Capture> captures, std::vector<std::uint32_t> rewrites) {
    const auto projected = segmented_prefill_units(begin, end, chunk, quantum,
        std::span<const Capture>(captures), std::span<const std::uint32_t>(rewrites));
    const auto actual = execution_units(begin, end, chunk, quantum, captures, rewrites,
                                         chunk != quantum);
    if (projected < actual || (chunk == quantum && projected != actual)) {
        std::cerr << "prefill " << begin << ':' << end << " chunk=" << chunk
                  << " quantum=" << quantum << " projected=" << projected
                  << " actual=" << actual << '\n';
        return false;
    }
    return true;
}
} // namespace

int main() {
    // The production failure: a rewrite at 512 retains the 1024 nominal boundary.
    if (!check(0, 1536, 1024, 1024, {}, {512})) { return 1; }
    // A capture yields early and restarts that boundary; conflating it with a rewrite overcounts.
    if (!check(0, 1536, 1024, 1024, {{512}}, {512})) { return 1; }
    if (!check(256, 1792, 1024, 1024, {}, {768})) { return 1; }
    if (!check(256, 1792, 1024, 1024, {{768}}, {768, 1280})) { return 1; }
    if (!check(0, 2048, 1024, 1024, {{1024}}, {1024, 2048})) { return 1; }
    if (!check(512, 512, 1024, 1024, {{512}}, {512})) { return 1; }
    // Non-divisible automatic chunk sizes and changed scheduler budgets.
    if (!check(0, 2304, 1152, 1024, {}, {512, 1536})) { return 1; }
    if (!check(256, 3072, 2048, 1024, {{1536}}, {768, 1536, 2048})) { return 1; }
    // Exhaust all switch sequences for short intervals, all single boundaries,
    // and capture/rewrite coincidences. Scale-independent arithmetic also covers
    // automatic sizes whose service quantum does not divide their execution chunk.
    for (std::uint32_t quantum = 1; quantum <= 4; ++quantum) {
        for (std::uint32_t chunk = quantum; chunk <= 6; ++chunk) {
            for (std::uint32_t begin = 0; begin <= 2; ++begin) {
                for (std::uint32_t end = begin; end <= begin + 9; ++end) {
                    for (std::uint32_t boundary = begin; boundary <= end; ++boundary) {
                        if (!check(begin, end, chunk, quantum, {}, {boundary}) ||
                            !check(begin, end, chunk, quantum, {{boundary}}, {boundary}) ||
                            !check(begin, end, chunk, quantum, {{boundary}}, {begin+1, end})) {
                            return 1;
                        }
                    }
                }
            }
        }
    }
    std::cout << "prefill projection matches fixed execution and bounds automatic execution\n";
}
