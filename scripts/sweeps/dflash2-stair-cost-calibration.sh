#!/usr/bin/env bash
# Linux RTX 4090 Stair-cost calibration.
# Keeps DFlash2 proposal physically fixed at K15 and forces each target-verification rung.
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"

model="${1:-${NINFER_DFLASH2_MODEL:-}}"
if [[ -z "$model" ]]; then
  if [[ -n "${NINFER_MODEL_DIR:-}" && -f "$NINFER_MODEL_DIR/qwen3_8_27b_dflash2.ninfer" ]]; then
    model="$NINFER_MODEL_DIR/qwen3_8_27b_dflash2.ninfer"
  else
    for candidate in "$repo/models/qwen3_8_27b_dflash2.ninfer" "$repo/scripts/models/qwen3_8_27b_dflash2.ninfer"; do
      [[ -f "$candidate" ]] && { model="$candidate"; break; }
    done
  fi
fi
[[ -n "$model" && -f "$model" ]] || { echo "Missing DFlash2 artifact" >&2; exit 1; }

if [[ -n "${NINFER_EXE:-}" ]]; then
  exe="$NINFER_EXE"
else
  exe=""
  if [[ -n "${NINFER_BUILD_DIR:-}" && -x "$NINFER_BUILD_DIR/apps/ninfer" ]]; then
    exe="$NINFER_BUILD_DIR/apps/ninfer"
  else
    for build in build-sm89 build-linux build-ninja build; do
      [[ -x "$repo/$build/apps/ninfer" ]] && { exe="$repo/$build/apps/ninfer"; break; }
    done
  fi
fi
[[ -x "$exe" ]] || { echo "ninfer executable not found" >&2; exit 1; }

out="${NINFER_SWEEP_OUT:-profiles/sweeps}"
mkdir -p "$out"

prompt="Explain why speculative decoding throughput depends on both accepted tokens per round and the non-linear cost of target verification width. Use a concrete CUDA inference example and discuss small-M kernel staircase effects."

printf "config,rep,decode_tok_s,generated_tokens,rounds,seconds_per_round,accepted,tok_per_round,sha256\n"
for spec in "rung3:1,100,100,100" "rung7:100,1,100,100" "rung11:100,100,1,100" "rung15:100,100,100,1"; do
  label="${spec%%:*}"
  costs="${spec#*:}"
  for rep in 1 2 3 4 5 6 7; do
    stem="$out/staircost_${label}_${rep}"
    stdout="$stem.txt"
    stderr="$stem.err.log"
    if ! "$exe" "$model" \
      --prompt "$prompt" --max-new 768 --max-context 8192 --kv-dtype int8 \
      --spec dflash2 --draft-tokens 15 \
      --spec-router stair --spec-stair-widths 3,7,11,15 \
      --spec-stair-costs "$costs" --spec-stair-draft-cost 0 \
      --spec-stair-warmup 0 --spec-stair-probe-period 0 --spec-stair-margin 0 \
      --greedy --no-thinking >"$stdout" 2>"$stderr"; then
      printf "%s,%s,FAILED,,,,,,\n" "$label" "$rep"
      tail -n 6 "$stderr" | sed "s/^/    /"
      continue
    fi

    decode="$(sed -nE "s/.*decode speed[[:space:]]+([0-9.]+) tok\/s.*/\1/p" "$stderr" | tail -n1)"
    generated="$(sed -nE "s/.*generated tokens[[:space:]]+([0-9]+).*/\1/p" "$stderr" | tail -n1)"
    rounds="$(sed -nE "s/.*[^[:space:]]+ rounds[[:space:]]+([0-9]+).*/\1/p" "$stderr" | tail -n1)"
    accepted="$(sed -nE "s/.*[^[:space:]]+ accepted tokens[[:space:]]+([0-9]+).*/\1/p" "$stderr" | tail -n1)"
    tpr="$(sed -nE "s/.*[^[:space:]]+ acceptance length[[:space:]]+([0-9.]+) tok\/round.*/\1/p" "$stderr" | tail -n1)"
    spr=""
    if [[ -n "$decode" && -n "$generated" && -n "$rounds" ]]; then
      spr="$(python3 - "$generated" "$decode" "$rounds" <<'PY'
import sys
g, d, r = map(float, sys.argv[1:])
print((g / d) / r if d > 0 and r > 0 else "")
PY
)"
    fi
    hash="$(sha256sum "$stdout" | awk "{print \$1}")"
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" "$label" "$rep" "$decode" "$generated" "$rounds" "$spr" "$accepted" "$tpr" "$hash"
  done
done
