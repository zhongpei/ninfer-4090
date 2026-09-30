#!/usr/bin/env python3
"""Install a trained DFlash2 checkpoint into a NInfer artifact using the existing converter.

This is a thin, reproducible wrapper around `python -m tools.convert`. It does not invent a new
artifact format or a second quantizer.

Examples:

  # Standard Qwen3.8 groupwise target + trained DFlash2
  python tools/export_dflash2_to_ninfer.py \
      --model /models/Qwen3.8-27B \
      --drafter train/ft-b16 \
      --recipe qwen3_8_27b \
      --out out/qwen3.8-27b-ft-b16.ninfer

  # Reuse the same custom/ternary conversion inputs as a production target
  python tools/export_dflash2_to_ninfer.py \
      --model /models/Qwen3.8-27B \
      --drafter train/ft-b16 \
      --recipe /path/to/production_recipe.py \
      --source quantized=/models/production-quant-source \
      --out out/qwen3.8-ft-b16.ninfer
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="primary target HF checkpoint/config")
    ap.add_argument("--drafter", required=True, help="trained DFlash2 HF-style checkpoint")
    ap.add_argument("--recipe", required=True, help="existing NInfer recipe or recipe.py[:function]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--components", default="text,dflash2")
    ap.add_argument("--source", action="append", default=[],
                    help="additional converter source NAME=PATH; repeatable")
    ap.add_argument("--override", default="")
    ap.add_argument("--proposal", action="store_true")
    ap.add_argument("--proposal-rows", type=int, default=131072)
    ap.add_argument("--ranking", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rows-per-chunk", type=int, default=512)
    ap.add_argument("--max-file-bytes", type=int, default=32_000_000_000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    drafter = os.path.abspath(os.path.expanduser(args.drafter))
    if not os.path.exists(os.path.join(drafter, "config.json")):
        raise SystemExit(f"{drafter}: trained drafter has no config.json")
    if not any(name.endswith(".safetensors") for name in os.listdir(drafter)):
        raise SystemExit(f"{drafter}: trained drafter has no safetensors")

    sources = list(args.source)
    if any(item.split("=", 1)[0] == "dflash2" for item in sources if "=" in item):
        raise SystemExit("--source dflash2=... is reserved; use --drafter")
    sources.append("dflash2=" + drafter)

    cmd = [
        sys.executable, "-m", "tools.convert",
        "--model", os.path.expanduser(args.model),
        "--recipe", args.recipe,
        "--components", args.components,
        "--out", os.path.expanduser(args.out),
        "--device", args.device,
        "--rows-per-chunk", str(args.rows_per_chunk),
        "--max-file-bytes", str(args.max_file_bytes),
    ]
    for source in sources:
        cmd += ["--source", source]
    if args.override:
        cmd += ["--override", args.override]
    if args.proposal:
        cmd += ["--proposal", "--proposal-rows", str(args.proposal_rows)]
    if args.ranking:
        cmd += ["--ranking", os.path.expanduser(args.ranking)]
    if args.name:
        cmd += ["--name", args.name]

    print(" ".join(shlex.quote(part) for part in cmd), flush=True)
    if args.dry_run:
        return
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
