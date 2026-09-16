# S1 RAG Standalone prior-plan: FP8

**Qwen3-Coder-30B-A3B-Instruct partial FP8**, H100 80 GB, PIE 0.4.0, driver **cuda_native**. Measured September 13, 2026.

runtime_quant=fp8 quantizes 192 attention projections only; MoE experts and KV remain BF16; configured activation dtype is BF16.

Ten S1 cases, 3 repetitions: **60 measured plans**, 6 discarded warmups. No runtime failures in the measured supported conditions; **0 length-capped outputs**. No accuracy evaluation.

| Condition | Decode tokens/s | ms/token | Mean plan decode | Mean output tokens | BF16 tokens/s | Rate vs BF16 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 32.03 | 31.22 | 4.29 s | 137.3 | 33.20 | -3.5% |
| speculated | 39.15 | 25.54 | 3.35 s | 131.3 | 40.66 | -3.7% |

speculated vs baseline: **1.22× throughput**, **-18.2% change in time per token**.

## Draft acceptance and output agreement

- speculated: 744/1407 drafts accepted (**52.9%**).
- Exact text agreement: **15/30** paired runs.

Output agreement is a verifier diagnostic, not a plan-accuracy score. Mismatching outputs mean exact output preservation is not established. Mean plan latency also depends on generated length; ms/token is the normalized comparison.

## Protocol

- Same compiled inferlet WASM and dataset hashes as the BF16 run, with greedy sampling, a 1,024-token cap, 512-token prefill chunks, eight-token drafts and two-token suffix matching. Case and mode/condition order rotate across repetitions.
- Prior plans are the first JSON queue from each task’s top-ranked retrieved example. This is a proxy for prior similar-request history; no current generated answer is used as a draft. Draft text is not added to the verifier prompt.
- The first sample is counted in prefill. Decode throughput counts subsequent sampled tokens including the final stop-token step, divided by decode time. Cache construction, prefill and token-to-text conversion are excluded. Requests are sequential.
- Standalone uses selected examples inline. The ablation uses the same contiguous cached catalog and decode loop across its supported conditions; masks select only indexed task examples. These are separate comparisons.
- BF16 comparisons use the saved earlier measurements. Q5_K_M uses the portable driver; comparing it with BF16/native CUDA includes a driver change and is not an isolated quantization experiment.

Raw measurements and provenance: `results/s1-prior-plan-a3b-partial-fp8-2026-09-13/`. Includes `results.jsonl`, `summary.json`, `metadata.json`, `config.toml`, invocation logs and `validation.json`.

[Saved measurements](../../../pie-runpod-setup/results/s1-prior-plan-a3b-partial-fp8-2026-09-13/summary.json)
