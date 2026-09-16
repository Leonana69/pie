# S1 RAG Standalone prior-plan: Q5_K_M

**Qwen3-Coder-30B-A3B-Instruct Q5_K_M**, H100 80 GB, PIE 0.4.0, driver **portable**. Measured September 13, 2026.

Q5_K_M GGUF weights; portable driver default unquantized KV (F16 storage in driver); configured activation dtype BF16. Cross-format comparisons also change driver.

Ten S1 cases, 3 repetitions: **60 measured plans**, 6 discarded warmups. No runtime failures in the measured supported conditions; **0 length-capped outputs**. No accuracy evaluation.

| Condition | Decode tokens/s | ms/token | Mean plan decode | Mean output tokens | BF16 tokens/s | Rate vs BF16 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 34.69 | 28.83 | 3.80 s | 131.9 | 33.20 | +4.5% |
| speculated | 38.39 | 26.05 | 3.54 s | 135.8 | 40.66 | -5.6% |

speculated vs baseline: **1.11× throughput**, **-9.6% change in time per token**.

## Draft acceptance and output agreement

- speculated: 555/1785 drafts accepted (**31.1%**).
- Exact text agreement: **27/30** paired runs.

Output agreement is a verifier diagnostic, not a plan-accuracy score. Mismatching outputs mean exact output preservation is not established. Mean plan latency also depends on generated length; ms/token is the normalized comparison.

## Protocol

- Same compiled inferlet WASM and dataset hashes as the BF16 run, with greedy sampling, a 1,024-token cap, 512-token prefill chunks, eight-token drafts and two-token suffix matching. Case and mode/condition order rotate across repetitions.
- Prior plans are the first JSON queue from each task’s top-ranked retrieved example. This is a proxy for prior similar-request history; no current generated answer is used as a draft. Draft text is not added to the verifier prompt.
- The first sample is counted in prefill. Decode throughput counts subsequent sampled tokens including the final stop-token step, divided by decode time. Cache construction, prefill and token-to-text conversion are excluded. Requests are sequential.
- Standalone uses selected examples inline. The ablation uses the same contiguous cached catalog and decode loop across its supported conditions; masks select only indexed task examples. These are separate comparisons.
- BF16 comparisons use the saved earlier measurements. Q5_K_M uses the portable driver; comparing it with BF16/native CUDA includes a driver change and is not an isolated quantization experiment.

## Q5_K_M support limit

The native load and masked-condition probes are saved separately. Unsupported conditions: Masking only, Prior plan + masking. Failed probes have no reported throughput; the table includes only successful supported conditions.
Support evidence: `results/s1-q5-support-2026-09-13/summary.json` and its raw logs.

Raw measurements and provenance: `results/s1-prior-plan-a3b-q5-k-m-2026-09-13/`. Includes `results.jsonl`, `summary.json`, `metadata.json`, `config.toml`, invocation logs and `validation.json`.

[Saved measurements](../../../pie-runpod-setup/results/s1-prior-plan-a3b-q5-k-m-2026-09-13/summary.json)
