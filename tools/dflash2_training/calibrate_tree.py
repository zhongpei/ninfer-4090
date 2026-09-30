#!/usr/bin/env python3
"""Score recorded DFlash2 lattices at multiple tree budgets.

Input is JSONL. Each row contains anchor,candidates,scores and optional target_tokens.  When target
tokens are present the script walks the tree exactly as a greedy target verifier would: from the
current node, follow the child equal to the target token until no child matches.

This is the offline half of TandemLLM's tree sweep and gives NInfer a reproducible way to decide
whether a future branch-aware verifier is worth its StateImage/ReplaySSM complexity.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from tools.dflash2_training.tree_policy import lattice_tree


def accepted(tree, target):
    children = defaultdict(dict)
    for node in range(1, len(tree.tokens)):
        children[tree.parents[node]][tree.tokens[node]] = node
    node = 0
    count = 0
    for token in target:
        nxt = children[node].get(int(token))
        if nxt is None:
            break
        count += 1
        node = nxt
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl")
    ap.add_argument("--budgets", default="3,7,11,15,23,31")
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()
    budgets = [int(x) for x in args.budgets.split(",") if x]

    totals = {b: [0, 0, 0] for b in budgets}  # accepted, rows, nodes
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            target = row.get("target_tokens", [])
            for budget in budgets:
                tree = lattice_tree(int(row["anchor"]), row["candidates"], row["scores"],
                                    budget, args.temperature)
                totals[budget][0] += accepted(tree, target)
                totals[budget][1] += 1
                totals[budget][2] += tree.n_draft

    print("budget,rows,mean_nodes,accepted_per_round")
    for budget in budgets:
        acc, rows, nodes = totals[budget]
        print(f"{budget},{rows},{nodes/max(rows,1):.3f},{acc/max(rows,1):.3f}")


if __name__ == "__main__":
    main()
