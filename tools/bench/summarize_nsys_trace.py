#!/usr/bin/env python3
"""Summarize native nsys CSV exports without running benchmarks or profiling.

Export cuda_gpu_trace and (for decode selection) nvtx_gpu_proj_trace with
`nsys stats --format csv`. Supply the exact projected NVTX Name as --range-name.
The selected window spans the first through last matching projected GPU range,
including gaps between rounds. A capture must contain one measured request;
separate requests cannot be distinguished by this window selection.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"{path}: expected native nsys CSV columns {sorted(required)}")
        rows = list(reader)
    if not rows or any(None in row or None in row.values() for row in rows):
        raise ValueError(f"{path}: empty or malformed CSV")
    return rows


def union_ns(intervals: list[tuple[int, int]]) -> int:
    total = 0
    end = -1
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def summarize(rows: list[dict[str, str]], window: tuple[int, int] | None = None) -> dict:
    events = [(int(row["Start (ns)"]), int(row["Duration (ns)"]), row) for row in rows]
    if any(start < 0 or duration <= 0 for start, duration, _ in events):
        raise ValueError("trace contains invalid GPU timestamps/durations")
    if window is None:
        window = min(start for start, _, _ in events), max(s + d for s, d, _ in events)
    low, high = window
    if low < 0 or high <= low:
        raise ValueError("selected window must have positive duration")
    selected = [(s, d, r) for s, d, r in events if s < high and s + d > low]
    if not selected:
        raise ValueError("no GPU events in selected window")
    if any(s < low or s + d > high for s, d, _ in selected):
        raise ValueError("selected boundary cuts a GPU event; choose a complete projected range")
    devices = sorted({r["Device"] for _, _, r in selected})
    contexts = sorted({r["Ctx"] for _, _, r in selected})
    if len(devices) != 1 or len(contexts) != 1:
        raise ValueError("selection spans multiple devices/contexts; export one inference process")
    groups = defaultdict(list)
    launch_fields = ("GrdX", "GrdY", "GrdZ", "BlkX", "BlkY", "BlkZ", "Reg/Trd")
    memory_fields = sorted(k for k in rows[0] if k.startswith(("StcSMem", "DymSMem")))
    for start, duration, row in selected:
        if row["GrdX"]:  # Memcpy/memset have no grid; retain them in total GPU busy time.
            key = tuple(row[k] for k in ("Name", *launch_fields, *memory_fields))
            groups[key].append(duration)
    kernels = []
    for key, durations in groups.items():
        kernels.append({"name": key[0], "launch": dict(zip((*launch_fields, *memory_fields), key[1:])),
                        "calls": len(durations), "total_ns": sum(durations),
                        "mean_ns": sum(durations) / len(durations),
                        "min_ns": min(durations), "max_ns": max(durations)})
    busy = union_ns([(s, s + d) for s, d, _ in selected])
    return {"schema_version": 1, "window_start_ns": low, "window_end_ns": high,
            "window_ns": high - low, "gpu_busy_union_ns": busy,
            "no_traced_gpu_work_ns": high - low - busy,
            "kernel_duration_sum_ns": sum(k["total_ns"] for k in kernels),
            "devices": devices, "contexts": contexts,
            "kernels": sorted(kernels, key=lambda k: k["total_ns"], reverse=True),
            "limitations": ["Kernel duration sums can overlap; GPU busy time uses their union including memops.",
                            "No traced GPU work is not automatically a CPU bottleneck.",
                            "NVTX GPU projection identifies launches, not Graph capture-time internal phases.",
                            "Profiler timings are not normal benchmark throughput."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-trace", type=Path, required=True)
    parser.add_argument("--ranges", type=Path, help="native nvtx_gpu_proj_trace CSV")
    parser.add_argument("--range-name", help="exact Name from projected range CSV")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.ranges) != bool(args.range_name):
        parser.error("--ranges and --range-name must be supplied together")
    try:
        window = None
        matched = []
        if args.ranges:
            ranges = read_csv(args.ranges, {"Name", "Projected Start (ns)", "Projected Duration (ns)"})
            matched = [r for r in ranges if r["Name"] == args.range_name]
            if not matched:
                raise ValueError(f"no projected ranges named {args.range_name!r}")
            window = (min(int(r["Projected Start (ns)"]) for r in matched),
                      max(int(r["Projected Start (ns)"]) + int(r["Projected Duration (ns)"])
                          for r in matched))
        result = summarize(read_csv(args.gpu_trace, {"Start (ns)", "Duration (ns)", "Name",
                                                   "GrdX", "Device", "Ctx"}), window)
        result.update({"gpu_trace": str(args.gpu_trace), "range_source": str(args.ranges) if args.ranges else None,
                       "range_name": args.range_name, "matched_ranges": len(matched)})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
