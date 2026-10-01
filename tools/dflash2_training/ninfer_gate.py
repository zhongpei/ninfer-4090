#!/usr/bin/env python3
"""Gate trained DFlash2 companions against the actual converted NInfer artifacts.

This tool runs the NInfer CLI itself, not the Hugging Face reference model.  It therefore catches
conversion/quantization/runtime regressions after a trained companion has been merged with a T2/Q5
or Ternary-Bonsai target.

Input JSONL accepts either a string or {"name":...,"text":...}.  Repeat --artifact LABEL=PATH to
compare the released drafter, a custom b16 checkpoint, or different converted targets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path


PATTERNS = {
    "decode_tok_s": re.compile(r"decode speed\s+([\d.]+) tok/s"),
    "generated": re.compile(r"generated tokens\s+(\d+)"),
    "rounds": re.compile(r"\S+ rounds\s+(\d+)"),
    "drafted": re.compile(r"\S+ drafted tokens\s+(\d+)"),
    "accepted": re.compile(r"\S+ accepted tokens\s+(\d+)"),
    "acceptance_length": re.compile(r"\S+ acceptance length\s+([\d.]+) tok/round"),
    "tree_rounds": re.compile(r"tree rounds\s+(\d+)"),
    "tree_fallback": re.compile(r"tree fallback rounds\s+(\d+)"),
    "tree_nodes": re.compile(r"tree nodes\s+(\d+)"),
    "tree_accepted": re.compile(r"tree accepted drafts\s+(\d+)"),
}


def parse_artifact(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--artifact must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("--artifact must be LABEL=PATH")
    return label, Path(path)


def prompts(path: Path):
    with path.open(encoding="utf-8") as f:
        for index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, str):
                yield f"prompt-{index:04d}", value
            elif isinstance(value, dict) and isinstance(value.get("text"), str):
                yield str(value.get("name", f"prompt-{index:04d}")), value["text"]
            else:
                raise ValueError(f"{path}:{index+1}: expected string or object with text")


def extract(stderr: str):
    out = {}
    for name, pattern in PATTERNS.items():
        match = pattern.search(stderr)
        out[name] = match.group(1) if match else ""
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ninfer", required=True, type=Path)
    ap.add_argument("--artifact", action="append", required=True, type=parse_artifact)
    ap.add_argument("--prompts", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--max-context", type=int, default=8192)
    ap.add_argument("--kv-dtype", default="int8")
    ap.add_argument("--draft-tokens", type=int, default=15)
    ap.add_argument("--tree", action="store_true")
    ap.add_argument("--extra", action="append", default=[],
                    help="one additional CLI argument; repeat for flags/values")
    args = ap.parse_args()

    prompt_rows = list(prompts(args.prompts))
    if not prompt_rows:
        raise SystemExit("no prompts")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "artifact","prompt","returncode","decode_tok_s","generated","rounds","drafted","accepted",
        "acceptance_length","tree_rounds","tree_fallback","tree_nodes","tree_accepted","sha256",
    ]
    rows = []
    for label, artifact in args.artifact:
        if not artifact.exists():
            raise SystemExit(f"missing artifact: {artifact}")
        for name, text in prompt_rows:
            command = [
                str(args.ninfer), str(artifact), "--prompt", text,
                "--max-new", str(args.max_new), "--max-context", str(args.max_context),
                "--kv-dtype", args.kv_dtype, "--spec", "dflash2",
                "--draft-tokens", str(args.draft_tokens), "--greedy", "--no-thinking",
            ]
            if args.tree:
                command += [
                    "--spec-tree", "lattice",
                    "--spec-tree-nodes", str(args.draft_tokens),
                    "--spec-tree-spine", str(min(7, args.draft_tokens)),
                ]
            command += args.extra
            completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout = completed.stdout
            stderr = completed.stderr.decode("utf-8", errors="replace")
            metrics = extract(stderr)
            row = {
                "artifact": label,
                "prompt": name,
                "returncode": completed.returncode,
                **metrics,
                "sha256": hashlib.sha256(stdout).hexdigest(),
            }
            rows.append(row)
            print(
                f"{label} {name}: rc={completed.returncode} "
                f"tok/s={metrics['decode_tok_s']} accepted={metrics['accepted']} "
                f"hash={row['sha256'][:12]}",
                flush=True,
            )

    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nSummary")
    for label, _ in args.artifact:
        selected = [r for r in rows if r["artifact"] == label and r["returncode"] == 0]
        speeds = [float(r["decode_tok_s"]) for r in selected if r["decode_tok_s"]]
        lengths = [float(r["acceptance_length"]) for r in selected if r["acceptance_length"]]
        failures = len([r for r in rows if r["artifact"] == label and r["returncode"] != 0])
        print(
            f"{label}: ok={len(selected)} fail={failures} "
            f"median_tok_s={statistics.median(speeds) if speeds else float('nan'):.3f} "
            f"median_tok_round={statistics.median(lengths) if lengths else float('nan'):.3f}"
        )

    # Cross-artifact prompt hashes make model-output changes explicit rather than silently mixing
    # quality and speed. They are diagnostic, not a requirement that differently trained drafters
    # produce byte-identical output under every numerical near-tie.
    by_prompt = {}
    for row in rows:
        if row["returncode"] == 0:
            by_prompt.setdefault(row["prompt"], set()).add(row["sha256"])
    changed = [name for name, hashes in by_prompt.items() if len(hashes) > 1]
    print(f"cross-artifact output-hash differences: {len(changed)}/{len(by_prompt)} prompts")
    if changed:
        print("  " + ", ".join(changed[:20]))

    if any(row["returncode"] != 0 for row in rows):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
