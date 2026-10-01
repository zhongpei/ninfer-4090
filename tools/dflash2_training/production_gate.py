#!/usr/bin/env python3
"""Production gate for a trained DFlash2 companion.

The gate is deliberately boring and reproducible:
  1. require machine-readable heldout metrics from train.py;
  2. enforce block-16 and operator-selected acceptance/improvement thresholds;
  3. invoke the existing NInfer converter;
  4. optionally compare greedy output bytes against a non-speculative baseline artifact.

A drafter that is fast but changes greedy target output is rejected. Sampling-tree correctness is
not inferred from this gate; positive-temperature tree verification remains an explicit runtime
fallback until it has an exact branch rejection sampler.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path


DEFAULT_PROMPTS = [
    "Explain paged KV cache fragmentation in three concise paragraphs.",
    "Write a Python function that merges overlapping integer intervals and explain its complexity.",
    "Compare recurrent state and full-attention KV when speculative branches are rejected.",
]


def read_metrics(drafter: Path) -> dict:
    path = drafter / "training_metrics.json"
    if not path.is_file():
        raise SystemExit(f"missing training metrics: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("block", "before", "final", "best"):
        if key not in data:
            raise SystemExit(f"training metrics missing {key}")
    return data


def gate_metrics(data: dict, min_best: float, min_improvement: float,
                 require_block: int) -> dict:
    block = int(data["block"])
    if require_block and block != require_block:
        raise SystemExit(f"training gate requires block={require_block}, got {block}")
    before = float(data["before"]["ALL"])
    final = float(data["final"]["ALL"])
    best = float(data["best"])
    if not all(math.isfinite(x) for x in (before, final, best)):
        raise SystemExit("training gate saw non-finite acceptance metric")
    if best < min_best:
        raise SystemExit(f"heldout best {best:.4f} < required {min_best:.4f}")
    improvement = best - before
    if improvement < min_improvement:
        raise SystemExit(
            f"heldout improvement {improvement:.4f} < required {min_improvement:.4f}")
    return {
        "block": block,
        "before": before,
        "final": final,
        "best": best,
        "improvement": improvement,
        "data_fingerprint": data.get("data_fingerprint", ""),
    }


def export(args) -> None:
    command = [
        sys.executable, "-m", "tools.dflash2_training.export_ninfer",
        "--model", args.model,
        "--drafter", str(args.drafter),
        "--out", str(args.out),
        "--device", args.device,
    ]
    if args.recipe:
        command += ["--recipe", args.recipe]
    if args.ternary:
        command += ["--ternary", args.ternary]
    if args.name:
        command += ["--name", args.name]
    if args.proposal:
        command += ["--proposal", "--proposal-rows", str(args.proposal_rows)]
    for source in args.source:
        command += ["--source", source]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def run_greedy(exe: Path, artifact: Path, prompt: str, max_new: int,
               speculative: bool) -> tuple[str, bytes]:
    command = [
        str(exe), str(artifact), "--prompt", prompt,
        "--max-new", str(max_new), "--greedy", "--no-thinking", "--raw-output",
    ]
    if speculative:
        command += [
            "--spec", "dflash2", "--draft-tokens", "15",
            "--spec-tree", "off",
        ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return hashlib.sha256(result.stdout).hexdigest(), result.stdout


def correctness_gate(args) -> list[dict]:
    if not args.ninfer and not args.baseline_artifact:
        return []
    if not args.ninfer or not args.baseline_artifact:
        raise SystemExit("--ninfer and --baseline-artifact must be supplied together")
    prompts = args.prompt or DEFAULT_PROMPTS
    records = []
    for index, prompt in enumerate(prompts):
        baseline_hash, baseline = run_greedy(
            args.ninfer, args.baseline_artifact, prompt, args.max_new, False)
        candidate_hash, candidate = run_greedy(
            args.ninfer, args.out, prompt, args.max_new, True)
        equal = baseline == candidate
        records.append({
            "index": index,
            "baseline_sha256": baseline_hash,
            "candidate_sha256": candidate_hash,
            "equal": equal,
        })
        print(f"greedy[{index}] baseline={baseline_hash} candidate={candidate_hash} equal={equal}")
        if not equal:
            raise SystemExit(f"greedy correctness gate failed on prompt {index}")
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="target/base HF checkpoint")
    ap.add_argument("--drafter", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-best", type=float, default=0.0,
                    help="minimum heldout accepted target tokens/block")
    ap.add_argument("--min-improvement", type=float, default=0.0,
                    help="minimum best-minus-before accepted tokens/block")
    ap.add_argument("--require-block", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--recipe", default="")
    ap.add_argument("--ternary", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--proposal", action="store_true")
    ap.add_argument("--proposal-rows", type=int, default=131072)
    ap.add_argument("--source", action="append", default=[])
    ap.add_argument("--ninfer", type=Path, default=None,
                    help="optional ninfer executable for greedy equivalence gate")
    ap.add_argument("--baseline-artifact", type=Path, default=None,
                    help="same target without speculative companion")
    ap.add_argument("--prompt", action="append", default=[])
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--manifest", type=Path, default=None)
    args = ap.parse_args()

    metrics = gate_metrics(
        read_metrics(args.drafter), args.min_best, args.min_improvement, args.require_block)
    print("training gate:", json.dumps(metrics, sort_keys=True), flush=True)
    export(args)
    if not args.out.is_file():
        raise SystemExit(f"converter did not produce artifact: {args.out}")
    artifact_sha = hashlib.sha256(args.out.read_bytes()).hexdigest()
    greedy = correctness_gate(args)

    manifest = {
        "version": 1,
        "artifact": str(args.out),
        "artifact_sha256": artifact_sha,
        "training_gate": metrics,
        "greedy_equivalence": greedy,
        "sampling_tree_gate": "not_claimed; runtime tree remains raw-greedy only",
    }
    manifest_path = args.manifest or args.out.with_suffix(args.out.suffix + ".gate.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"production gate passed -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
