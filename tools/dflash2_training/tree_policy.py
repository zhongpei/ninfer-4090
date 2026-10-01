#!/usr/bin/env python3
"""DFlash2 lattice -> speculative tree planner.

Adapted from TandemLLM engine/tree.py.  This is intentionally pure Python: it lets training and
offline calibration compare chain, greedy-spine and best-first tree budgets without changing the
NInfer runtime.  Runtime tree verification needs branch-aware target GDN/ReplaySSM state and is a
separate engine change.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import dataclass, field


@dataclass
class DraftTree:
    tokens: list[int]
    parents: list[int]
    scores: list[float]
    sources: list[str]

    @property
    def n_draft(self):
        return max(0, len(self.tokens) - 1)

    def depths(self):
        out = [0] * len(self.tokens)
        for i in range(1, len(self.tokens)):
            out[i] = out[self.parents[i]] + 1
        return out

    def check(self):
        if not self.tokens or len(self.tokens) != len(self.parents):
            raise ValueError("invalid tree vector lengths")
        if self.parents[0] != -1:
            raise ValueError("root parent must be -1")
        for i, parent in enumerate(self.parents[1:], 1):
            if parent < 0 or parent >= i:
                raise ValueError(f"node {i} has invalid parent {parent}")


class Builder:
    def __init__(self, anchor: int):
        self.tokens = [anchor]
        self.parents = [-1]
        self.scores = [1.0]
        self.sources = ["anchor"]
        self.children: list[dict[int, int]] = [{}]

    def add(self, parent: int, token: int, score: float, source: str) -> int:
        existing = self.children[parent].get(token)
        if existing is not None:
            if score > self.scores[existing]:
                self.scores[existing] = score
            return existing
        node = len(self.tokens)
        self.tokens.append(int(token))
        self.parents.append(parent)
        self.scores.append(float(score))
        self.sources.append(source)
        self.children.append({})
        self.children[parent][int(token)] = node
        return node

    def build(self) -> DraftTree:
        # DFS preorder keeps every parent before its children and makes the runtime representation
        # compact enough for a future row-parent array.
        order: list[int] = []
        def walk(node: int):
            order.append(node)
            for child in sorted(self.children[node].values(),
                                key=lambda n: -self.scores[n]):
                walk(child)
        walk(0)
        remap = {old: new for new, old in enumerate(order)}
        tree = DraftTree(
            tokens=[self.tokens[i] for i in order],
            parents=[-1 if i == 0 else remap[self.parents[i]] for i in order],
            scores=[self.scores[i] for i in order],
            sources=[self.sources[i] for i in order],
        )
        tree.check()
        return tree


def log_softmax_row(values: list[float], temperature: float) -> list[float]:
    scaled = [v / temperature for v in values]
    maximum = max(scaled)
    z = maximum + math.log(sum(math.exp(v - maximum) for v in scaled))
    return [v - z for v in scaled]


def greedy_walk(scores: list[list[list[float]]]) -> list[int]:
    if not scores:
        return []
    row = 0
    result = []
    for layer, table in enumerate(scores):
        current = table[row]
        choice = max(range(len(current)), key=current.__getitem__)
        result.append(choice)
        row = choice
    return result


def lattice_tree(anchor: int, candidates: list[list[int]], scores: list[list[list[float]]],
                 budget: int, temperature: float = 1.0) -> DraftTree:
    """Greedy spine first, then highest path-probability alternatives.

    budget counts drafted nodes and excludes the anchor.
    """
    if budget <= 0 or not candidates:
        return Builder(anchor).build()
    logp = [[log_softmax_row(row, temperature) for row in table] for table in scores]
    greedy = greedy_walk(scores)

    builder = Builder(anchor)
    frontier = []
    tie = 0
    parent = 0
    row = 0
    path_lp = 0.0
    used = 0

    for slot, choice in enumerate(greedy):
        if used >= budget:
            break
        base = path_lp
        path_lp += logp[slot][row][choice]
        here = parent
        parent = builder.add(parent, candidates[slot][choice], math.exp(path_lp), "df2-greedy")
        used += 1
        for alt in range(len(candidates[slot])):
            if alt == choice:
                continue
            tie += 1
            heapq.heappush(frontier, (-(base + logp[slot][row][alt]), tie,
                                      here, slot, alt))
        row = choice

    while frontier and used < budget:
        negp, _, parent_id, slot, choice = heapq.heappop(frontier)
        lp = -negp
        node = builder.add(parent_id, candidates[slot][choice], math.exp(lp), "df2-tree")
        used += 1
        if slot + 1 < len(candidates):
            for nxt in range(len(candidates[slot + 1])):
                tie += 1
                heapq.heappush(frontier, (-(lp + logp[slot + 1][choice][nxt]), tie,
                                          node, slot + 1, nxt))
    return builder.build()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("lattice", help="JSON with anchor,candidates,scores")
    ap.add_argument("--budget", type=int, default=15)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    with open(args.lattice, encoding="utf-8") as f:
        value = json.load(f)
    tree = lattice_tree(
        int(value["anchor"]), value["candidates"], value["scores"],
        args.budget, args.temperature)
    result = {
        "tokens": tree.tokens,
        "parents": tree.parents,
        "scores": tree.scores,
        "sources": tree.sources,
        "depths": tree.depths(),
    }
    text = json.dumps(result, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
