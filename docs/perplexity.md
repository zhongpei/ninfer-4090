# Perplexity evaluation

`ninfer-perplexity` measures the causal perplexity produced by a v3 `.ninfer` artifact.
It uses the artifact's tokenizer, Text model, selected Main KV representation, final normalization,
and main output head. It is an offline evaluator, not a serving endpoint or a logits-export API.
Only Text weights and resources are loaded; Vision and speculative components are not required.

## Run the fixed corpus

The repository includes `ninfer-ppl-1m-v1`, a fixed set of 16 independent UTF-8 streams covering
English reference text, English long-form text, Chinese reference text, and NInfer C++/CUDA code.
`full` selects all streams; `--quick` selects one stream from each domain.

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b_nvfp4.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json \
  --quick \
  --kv-dtype int8
```

The default evaluation uses a 4,096-token context and a 2,048-token stride. Use `--context` and
`--stride` to change that protocol, or score one UTF-8 file with `--text FILE`. The available Main
KV representations are `bf16`, `int8`, `fp8`, `rk8v4`, `rk4v4`, `rk4v4-e8`,
`rk2v4-e8`, `nvfp4`, and `k8v4`.

Historical fixed-window measurements for selected formats are in
[`docs/config-calculator.html`](config-calculator.html). They do not qualify the new sm89 native
FP8 arithmetic or every format listed above on Bonsai 27B.

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b.ninfer \
  --text notes.txt \
  --context 16384 --stride 8192 \
  --kv-dtype int8
```

Run `./build/apps/ninfer-perplexity --help` for the complete command surface. The evaluator loads
the model once, reads and tokenizes every selected stream before scoring, and writes readable
startup, corpus, scoring, and per-stream summaries to stderr. Interactive weight loading and
scoring use one transient progress line; redirected scoring emits persistent progress every ten
seconds. `--log-level debug` exposes internal startup and stream-begin detail. The final
domain/overall table remains product output on stdout; the independent full-precision machine
report is `report.json` under `profiles/perplexity/` unless `--output` supplies an empty directory.

For KV-format comparisons, the recommended long-context profile is the full corpus with
`--context 65536 --stride 32768` and without `--quick`.

## Long-history KV quality

The ordinary fixed-window protocol answers "what is the model's perplexity with at most N tokens of
local context?" It is **not** a good test for accumulated KV-cache quantization drift because every
window starts with an empty cache.

Use `--depths` to measure the different question. For a requested prefix depth `D` the evaluator
materializes the complete token prefix `[0,D)` in the selected KV representation and scores only
the next `--tail` tokens:

```text
0 ------------------------------------------------ D -------- D+tail
|             real encoded KV history             |  scored  |
```

For example:

```bash
./build/apps/ninfer-perplexity models/Ternary-Bonsai-2-27B.ninfer \
  --text /absolute/path/long-evaluation-stream.txt \
  --kv-dtype int8 \
  --depths 8192,32768,65536,131072,196608,258048 \
  --tail 2048 \
  --output profiles/perplexity/bonsai-int8-depth
```

The bundled corpus streams are approximately 64K tokens each, with exact lengths determined by
the artifact tokenizer. They cannot individually cover 128K–258K prefix depths. To test those
depths, provide an explicitly constructed longer UTF-8 stream with `--text`, and record its source
order and construction in the test report. Do not infer long-depth coverage from the manifest
name.

Depths beyond an individual corpus stream are omitted for that stream and every report records the
exact stream/token coverage. The report schema is v3 and includes a `depths` table with token-weighted
NLL/PPL for each prefix depth.

For a complete KV A/B, use the matrix runner:

```bash
python3 -m tools.bench.run_kv_long_context_perplexity \
  --exe ./build/apps/ninfer-perplexity \
  --model models/Ternary-Bonsai-2-27B.ninfer \
  --text /absolute/path/long-evaluation-stream.txt \
  --out profiles/perplexity/bonsai-kv-depth
```

Its default arms are `int8,fp8,rk8v4,rk4v4-e8`. INT8 is the baseline. At every common depth the
summary reports `delta_mean_nll_vs_baseline`, `ppl_ratio_vs_baseline`, and percentage PPL change.
Positive values mean the encoded long-history cache made next-token likelihood worse.

This protocol deliberately scores only a suffix at each depth. Do not merge its "overall" number
with the ordinary full-corpus fixed-window PPL table; the useful result is the **per-depth curve**.

## Metric

In fixed-window mode, for a stream `x[0..N)`, every token after `x[0]` is scored exactly once. A window `[b,e)` with target
suffix `[s,e)` contributes:

```text
log p(x[i] | x[b], ..., x[i-1])  for i in [s,e)
```

Each window starts from empty State and Main KV, so history before `b` is deliberately excluded.
The reported metric is therefore fixed-window, truncated-context causal perplexity:

```text
mean_nll = -sum(logprob) / scored_tokens
perplexity = exp(mean_nll)
```

The first window scores `[1,min(context,N))`. Each later window advances by `stride` targets while
retaining up to `context-stride` preceding tokens as local context. Streams never share history.

## Comparing runs

For a numerical comparison, keep the corpus, context, stride, and execution settings fixed except
the variable being measured. Compare KV formats with the same artifact and weight formats with the
same KV format.

The corpus name is a workload scale, not an exact token count. Exact input and scored-token counts
are runtime results from the current artifact tokenizer and are recorded in each report. Reports
contain unrounded NLL/PPL values for every window, stream, domain, and the token-weighted overall
aggregate.

The schema-v3 report identifies the artifact's architecture, public name, actual weight formats
and prefill signature alongside the workload and numerical results.
