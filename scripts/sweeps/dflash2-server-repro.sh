#!/usr/bin/env bash
# Isolate the 2026-10-02 baseline-repeat failure on Linux. No production service is touched.
# Usage: bash scripts/sweeps/dflash2-server-repro.sh MODEL.ninfer [NEW_OUTPUT_DIRECTORY]
# Optional: NINFER_SERVE_EXE, NINFER_BUILD_DIR, NINFER_PYTHON, CUDA_VISIBLE_DEVICES.
# These are correctness/attribution experiments, not qualified performance measurements.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 MODEL.ninfer [NEW_OUTPUT_DIRECTORY]" >&2
  exit 2
fi
caller="$PWD"
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
model="$1"
[[ "$model" = /* ]] || model="$caller/$model"
[[ -f "$model" ]] || { echo "Missing model: $model" >&2; exit 1; }
python="${NINFER_PYTHON:-python3.11}"
if [[ "$python" = */* && "$python" != /* ]]; then python="$caller/$python"; fi
command -v "$python" >/dev/null || { echo "Missing Python: $python; set NINFER_PYTHON" >&2; exit 1; }
serve="${NINFER_SERVE_EXE:-}"
if [[ -n "$serve" && "$serve" != /* ]]; then serve="$caller/$serve"; fi
if [[ -z "$serve" && -n "${NINFER_BUILD_DIR:-}" ]]; then
  build="$NINFER_BUILD_DIR"
  [[ "$build" = /* ]] || build="$caller/$build"
  serve="$build/apps/ninfer-serve"
elif [[ -z "$serve" ]]; then
  for build in build-sm89 build-linux build-ninja build; do
    [[ -x "$repo/$build/apps/ninfer-serve" ]] && { serve="$repo/$build/apps/ninfer-serve"; break; }
  done
fi
[[ -x "$serve" ]] || { echo "Set NINFER_SERVE_EXE to the tested Linux binary" >&2; exit 1; }
if [[ $# -eq 2 ]]; then
  out="$2"
  [[ "$out" = /* ]] || out="$caller/$out"
  # Refuse even an existing empty path: this wrapper owns the whole experiment directory.
  [[ ! -e "$out" ]] || { echo "Output already exists: $out" >&2; exit 1; }
  mkdir -p -- "$out"
else
  mkdir -p -- "$repo/profiles"
  out="$(mktemp -d "$repo/profiles/server-repro-XXXXXXXX")"
fi
cd "$repo"
status=0
run_case() {
  local name="$1"
  shift
  local rc=0
  echo "[repro] $name"
  "$python" -m tools.dflash2_training.server_ab \
    --serve "$serve" --model "$model" --out "$out/$name" \
    --kv-dtype int8 --max-context 32768 --kv-capacity 32768 \
    --max-new 512 --thinking off --workloads prose,long-context \
    --pairs 1 --discard 0 --repeats 4 --cooldown 0 "$@" || rc=$?
  printf '%s\t%s\n' "$name" "$rc" >> "$out/exit-status.tsv"
  # A correctness failure must not hide the other isolation conditions. An interrupt stops all.
  if [[ "$rc" -eq 130 || "$rc" -eq 143 ]]; then exit "$rc"; fi
  if [[ "$rc" -ne 0 ]]; then status=2; fi
}

# Mixed short/long jobs fill C8 and expose changes when shorter requests leave the decode batch.
run_case baseline-production --arms baseline --concurrency 1,2,4,8 --server-capacity 8
run_case baseline-no-prefix --arms baseline --concurrency 1,2,4,8 --server-capacity 8 --prefix-reuse off
run_case baseline-no-graph --arms baseline --concurrency 1,2,4,8 --server-capacity 8 --cuda-graph off
run_case baseline-no-prefix-no-graph --arms baseline --concurrency 1,2,4,8 --server-capacity 8 --prefix-reuse off --cuda-graph off
# Startup capacity is distinct from current client concurrency; do not pool these configurations.
run_case baseline-capacity1 --arms baseline --concurrency 1 --server-capacity 1
# Restore production settings for the candidate gate; comparator closure includes target-only.
run_case candidates-production --arms dflash2-k15,tree15,tree15-stair,lookup-skip --concurrency 1,2,4,8 --server-capacity 8

echo "Diagnostic results: $out (status=$status; one pair is insufficient for a speedup claim)"
exit "$status"
