"""Run fixed prefill-chunk rungs plus VRAM-aware auto through ninfer_bench.

This is an isolated single-request prefill qualification. It measures kernel/chunk throughput and
records the chunk that Engine actually resolved for auto. It does not reproduce a serving Engine's
context-cache reservation or the decode/prefill interleave path; qualify those separately before
changing a production default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

DEFAULT_RUNGS = ("1024", "1536", "2048", "3072", "4096", "6144", "8192", "auto")


def parse_csv(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items or len(items) != len(set(items)):
        raise argparse.ArgumentTypeError("nonempty unique comma-separated values required")
    return items


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--prompts", default="1024,4096,16384,32768",
                        help="ninfer_bench -p list")
    parser.add_argument("--rungs", type=parse_csv, default=list(DEFAULT_RUNGS))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--kv-dtype", default="int8")
    parser.add_argument("--spec", choices=("none", "mtp", "dflash", "dflash2"), default="dflash2")
    parser.add_argument("--draft-tokens", type=int, default=7)
    parser.add_argument("--prefill-cublas", action="store_true")
    args = parser.parse_args(argv)

    if not args.exe.is_file() or not args.model.is_file():
        parser.error("--exe and --model must be existing files")
    if args.out.exists():
        parser.error("--out must be a new directory")
    if args.device < 0 or args.repetitions < 2 or args.warmup < 0 or args.max_context < 1024:
        parser.error("invalid device/repetition/warmup/context")
    allowed = set(DEFAULT_RUNGS)
    if any(rung not in allowed for rung in args.rungs):
        parser.error("rungs must be selected from " + ",".join(DEFAULT_RUNGS))

    args.out.mkdir(parents=True)
    arms = []
    for order, rung in enumerate(args.rungs):
        report = args.out / f"{order:02d}-{rung}.json"
        command = [
            str(args.exe.resolve()),
            "--weights", str(args.model.resolve()),
            "--device", str(args.device),
            "--max-ctx", str(args.max_context),
            "-p", args.prompts,
            "-r", str(args.repetitions),
            "--warmup", str(args.warmup),
            "--kv-dtype", args.kv_dtype,
            "--prefill-chunk", rung,
            "--output", "json",
            "--output-file", str(report),
        ]
        if args.spec != "none":
            command += ["--spec", args.spec, "--draft-tokens", str(args.draft_tokens)]
        if args.prefill_cublas:
            command.append("--prefill-cublas")
        run_dir = args.out / f"{order:02d}-{rung}"
        run_dir.mkdir()
        (run_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        with (run_dir / "stdout.log").open("wb") as stdout, (run_dir / "stderr.log").open("wb") as stderr:
            completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        if completed.returncode != 0 or not report.is_file():
            raise RuntimeError(f"prefill rung {rung} failed; see {run_dir}")
        raw = json.loads(report.read_text())
        config = raw["config"]
        tests = raw["tests"]
        arms.append({
            "requested": rung,
            "auto": bool(config.get("prefill_chunk_auto", False)),
            "resolved_chunk": int(config["prefill_chunk"]),
            "runtime_reservation_bytes": int(raw["memory"]["runtime_reservation_bytes"]),
            "workspace_capacity_bytes": int(raw["memory"]["workspace"]["capacity_bytes"]),
            "tests": [{
                "label": test["label"],
                "n_prompt": test["n_prompt"],
                "prefill_tok_s_mean": test["prefill_tok_s_mean"],
                "prefill_tok_s_stddev": test["prefill_tok_s_stddev"],
            } for test in tests],
            "report": str(report.resolve()),
        })

    labels = [test["label"] for test in arms[0]["tests"]]
    winners = {}
    for label in labels:
        candidates = []
        for arm in arms:
            row = next(test for test in arm["tests"] if test["label"] == label)
            if row["prefill_tok_s_mean"] is not None:
                candidates.append((float(row["prefill_tok_s_mean"]), arm["requested"],
                                   arm["resolved_chunk"]))
        if candidates:
            speed, requested, resolved = max(candidates)
            winners[label] = {
                "requested": requested,
                "resolved_chunk": resolved,
                "prefill_tok_s_mean": speed,
            }

    summary = {
        "schema_version": 1,
        "artifact_type": "ninfer_prefill_chunk_matrix",
        "scope": "isolated single-request ninfer_bench; not serving-cache/interleave qualification",
        "model": str(args.model.resolve()),
        "device": args.device,
        "max_context": args.max_context,
        "rungs": arms,
        "observed_fastest": winners,
    }
    path = args.out / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
