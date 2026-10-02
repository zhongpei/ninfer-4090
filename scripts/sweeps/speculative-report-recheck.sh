#!/usr/bin/env bash
# Linux qualification of the 2026-10-02 report fixes. No deployment/CI mutation.
set -euo pipefail
set -o noclobber
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
mode="${1:-aa}"
case "$mode" in aa|ab|cli|direct) ;; *) echo "usage: bash $0 [aa|ab|cli|direct] [MODEL.ninfer]" >&2; exit 2;; esac
model="${2:-${NINFER_DFLASH2_MODEL:-}}"
[[ -n "$model" && -f "$model" ]] || { echo "Set NINFER_DFLASH2_MODEL or pass an existing model as arg2." >&2; exit 2; }
python="${NINFER_PYTHON:-python3}"
command -v "$python" >/dev/null || { echo "Python not found: $python" >&2; exit 2; }
find_app() {
  local name="$1" override="$2" build
  if [[ -n "$override" ]]; then
    [[ -x "$override" ]] || { echo "Not executable: $override" >&2; return 2; }
    printf '%s\n' "$override"; return
  fi
  if [[ -n "${NINFER_BUILD_DIR:-}" ]]; then
    [[ -x "$NINFER_BUILD_DIR/apps/$name" ]] || { echo "Missing $NINFER_BUILD_DIR/apps/$name" >&2; return 2; }
    printf '%s\n' "$NINFER_BUILD_DIR/apps/$name"; return
  fi
  for build in build-sm89 build-linux build-ninja build; do
    [[ ! -x "$repo/$build/apps/$name" ]] || { printf '%s\n' "$repo/$build/apps/$name"; return; }
  done
  echo "Build $name or set NINFER_BUILD_DIR / executable override." >&2; return 2
}
out="${NINFER_AB_OUT:-profiles/report-recheck/$(date -u +%Y%m%dT%H%M%S)-$$}"
mkdir -p "$out"
# Explicit snapshots, not synthetic assertions about exclusivity or locked clocks.
{
  date -u +%Y-%m-%dT%H:%M:%SZ
  uname -a
  "$python" --version
  git rev-parse HEAD 2>/dev/null || true
  git status --short 2>/dev/null || true
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
  if command -v nvidia-smi >/dev/null; then nvidia-smi 2>&1 || true; fi
} > "$out/environment-$mode.txt"
if [[ "$mode" == cli ]]; then
  exe="$(find_app ninfer "${NINFER_EXE:-}")"
  exec "$python" -m tools.dflash2_training.ab_suite \
    --exe "$exe" --model "$model" --out "$out/cli" \
    --arms "${NINFER_ARMS:-baseline,dflash2-k7,dflash2-k11,dflash2-k15,lookup-skip}" \
    --workloads "${NINFER_WORKLOADS:-all}" --pairs "${NINFER_PAIRS:-4}" --discard 1 \
    --cooldown "${NINFER_COOLDOWN:-2}" --kv-dtype "${NINFER_KV_DTYPE:-int8}" --compare-baseline
fi
serve="$(find_app ninfer-serve "${NINFER_SERVE_EXE:-}")"
extra=()
[[ "$mode" != aa ]] || extra+=(--aa)
[[ "$mode" != direct ]] || extra+=(--compare-baseline)
exec "$python" -m tools.dflash2_training.server_ab \
  --serve "$serve" --model "$model" --out "$out/$mode" \
  --arms "${NINFER_ARMS:-baseline,dflash2-k15,tree15,tree15-stair,lookup-skip}" \
  --workloads "${NINFER_WORKLOADS:-all}" --concurrency "${NINFER_CONCURRENCY:-1,2,4,8}" \
  --engine-concurrency 8 --max-context 32768 --kv-capacity 32768 --max-tokens 512 \
  --pairs "${NINFER_PAIRS:-4}" --discard 1 --repeats 2 --cooldown "${NINFER_COOLDOWN:-2}" \
  --thinking "${NINFER_THINKING:-off}" --cache-mode "${NINFER_CACHE_MODE:-enabled}" \
  --kv-dtype "${NINFER_KV_DTYPE:-int8}" --port "${NINFER_AB_PORT:-18080}" "${extra[@]}"
