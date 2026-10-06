./build/apps/ninfer-serve ./Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --model-id qwen3.5-35b-a3b  --host 0.0.0.0 --port 8001 --device 0 \
  --max-context 262144 --kv-capacity auto --max-concurrency 8 \
  --prefill-chunk 1024 --kv-dtype int8 \
  --vision --vision-residency overlay --vision-max-merged 16384 \
  --spec dflash2 --draft-tokens 7 --spec-router fixed \
  --device-state-slots 0 --host-state-slots 64 --host-kv-mib 65536 \
  --max-private-continuations 16 --max-shared-prefixes 16 \
  --max-long-anchors-per-continuation 2 --max-cache-markers-per-request 4 \
  --auto-prefix-grid --request-log-jsonl profiles/bench/serve-cache.jsonl \
  --greedy --presence-penalty 0 --frequency-penalty 0 \
  --default-max-tokens 512
  # --no-thinking --default-max-tokens 512
