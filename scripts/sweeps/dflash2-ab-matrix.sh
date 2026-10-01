#!/usr/bin/env bash
# Linux entry point for the DFlash2/speculative-decoding A/B matrix.
#
# Profiles: smoke | core | full | server
#
# Environment:
#   NINFER_DFLASH2_MODEL  explicit .ninfer artifact
#   NINFER_MODEL_DIR      artifact directory fallback
#   NINFER_BUILD_DIR      build tree containing apps/ninfer and apps/ninfer-serve
#   NINFER_EXE            explicit CLI executable
#   NINFER_SERVE_EXE      explicit server executable
#   NINFER_AB_OUT         output directory (default profiles/ab-suite)
#   NINFER_PYTHON         Python executable (default python3)
#   CUDA_VISIBLE_DEVICES  standard CUDA device selection
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"

profile="${1:-core}"
case "$profile" in
  smoke|core|full|server) ;;
  *)
    echo "usage: $0 [smoke|core|full|server] [MODEL.ninfer]" >&2
    exit 2
    ;;
esac

model="${2:-${NINFER_DFLASH2_MODEL:-}}"
if [[ -z "$model" ]]; then
  if [[ -n "${NINFER_MODEL_DIR:-}" && -f "$NINFER_MODEL_DIR/qwen3_8_27b_dflash2.ninfer" ]]; then
    model="$NINFER_MODEL_DIR/qwen3_8_27b_dflash2.ninfer"
  else
    for candidate in "$repo/models/qwen3_8_27b_dflash2.ninfer" "$repo/scripts/models/qwen3_8_27b_dflash2.ninfer"; do
      if [[ -f "$candidate" ]]; then model="$candidate"; break; fi
    done
  fi
fi
if [[ -z "$model" || ! -f "$model" ]]; then
  echo "DFlash2 artifact not found. Pass it as arg2 or set NINFER_DFLASH2_MODEL/NINFER_MODEL_DIR." >&2
  exit 1
fi

find_app() {
  local app="$1"
  local explicit="$2"
  if [[ -n "$explicit" ]]; then
    [[ -x "$explicit" ]] || { echo "$app is not executable: $explicit" >&2; return 1; }
    printf "%s\n" "$explicit"
    return
  fi
  local candidate
  if [[ -n "${NINFER_BUILD_DIR:-}" ]]; then
    candidate="$NINFER_BUILD_DIR/apps/$app"
    [[ -x "$candidate" ]] && { printf "%s\n" "$candidate"; return; }
  fi
  for build in build-sm89 build-linux build-ninja build; do
    candidate="$repo/$build/apps/$app"
    [[ -x "$candidate" ]] && { printf "%s\n" "$candidate"; return; }
  done
  echo "$app not found; set NINFER_BUILD_DIR or explicit NINFER_EXE/NINFER_SERVE_EXE." >&2
  return 1
}

exe="$(find_app ninfer "${NINFER_EXE:-}")"
serve="$(find_app ninfer-serve "${NINFER_SERVE_EXE:-}")"
python="${NINFER_PYTHON:-python3}"
command -v "$python" >/dev/null 2>&1 || { echo "Python not found: $python" >&2; exit 1; }

out_root="${NINFER_AB_OUT:-profiles/ab-suite}"
mkdir -p "$out_root"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname 2>/dev/null || true)"
  echo "kernel=$(uname -srmo 2>/dev/null || true)"
  echo "python=$($python --version 2>&1 || true)"
  echo "exe=$exe"
  echo "serve=$serve"
  echo "model=$model"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,pstate,temperature.gpu,power.limit --format=csv,noheader 2>/dev/null || true
  fi
} > "$out_root/environment.txt"

run_cli() {
  local name="$1" kv="$2" arms="$3" workloads="$4" pairs="$5" discard="$6" cooldown="$7"
  "$python" -m tools.dflash2_training.ab_suite \
    --exe "$exe" --model "$model" --out "$out_root/$name" --kv-dtype "$kv" \
    --arms "$arms" --workloads "$workloads" --pairs "$pairs" --discard "$discard" --cooldown "$cooldown"
}

run_server() {
  local name="$1" kv="$2" arms="$3" workloads="$4"
  "$python" -m tools.dflash2_training.server_ab \
    --serve "$serve" --model "$model" --out "$out_root/$name" --kv-dtype "$kv" \
    --arms "$arms" --workloads "$workloads" --concurrency 1,2,4,8 --repeats 2 --cooldown 8
}

case "$profile" in
  smoke)
    run_cli smoke-int8 int8 baseline,dflash2-k15,tree15,tree15-stair,lookup-skip prose,code,lookup-repeat 2 0 1
    ;;
  core)
    run_cli core-int8 int8 all all 4 1 5
    for kv in fp8 rk8v4; do
      run_cli "kv-$kv" "$kv" baseline,dflash2-k15,tree15,tree15-stair prose,code,long-context 4 1 5
    done
    run_server server-int8 int8 baseline,dflash2-k15,tree15,tree15-stair,lookup-skip prose,chat,reasoning,code,lookup-repeat
    ;;
  full)
    run_cli full-int8 int8 all all 6 1 10
    for kv in fp8 rk8v4; do
      run_cli "full-$kv" "$kv" all prose,chat,reasoning,code,structured,lookup-repeat,long-context 5 1 8
    done
    for kv in int8 fp8 rk8v4; do
      run_server "server-$kv" "$kv" baseline,dflash2-k15,tree15,tree15-stair,lookup-skip prose,chat,reasoning,code,lookup-repeat
    done
    ;;
  server)
    run_server server-int8 int8 baseline,dflash2-k15,tree15,tree15-stair,lookup-skip prose,chat,reasoning,code,lookup-repeat
    ;;
esac

echo "A/B profile '$profile' complete: $out_root"
