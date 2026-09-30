#!/usr/bin/env python3
"""Convert a trained DFlash2 companion checkpoint into a NInfer artifact.

This is a thin wrapper over the repository's existing converter.  The trained directory remains a
normal DFlash2 source and can also be passed to tools.convert manually as --source dflash2=PATH.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="target/base HF checkpoint")
    ap.add_argument("--drafter", required=True, help="trained DFlash2 companion directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--recipe", default="qwen3_8_27b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--components", default="text,dflash2")
    ap.add_argument("--name", default="")
    ap.add_argument("--proposal", action="store_true")
    ap.add_argument("--proposal-rows", type=int, default=131072)
    ap.add_argument("--ranking", default="")
    ap.add_argument("--source", action="append", default=[],
                    help="additional NAME=PATH source forwarded to tools.convert")
    args = ap.parse_args()

    command = [
        sys.executable, "-m", "tools.convert",
        "--model", args.model,
        "--recipe", args.recipe,
        "--source", f"dflash2={args.drafter}",
        "--components", args.components,
        "--out", args.out,
        "--device", args.device,
    ]
    for source in args.source:
        command += ["--source", source]
    if args.name:
        command += ["--name", args.name]
    if args.proposal:
        command += ["--proposal", "--proposal-rows", str(args.proposal_rows)]
        if args.ranking:
            command += ["--ranking", args.ranking]

    print(" ".join(command), flush=True)
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
