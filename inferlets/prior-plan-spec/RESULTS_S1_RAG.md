# S1 RAG prior-plan results

Measured September 13, 2026 on **Qwen3-Coder-30B-A3B-Instruct BF16**, **H100 80 GB**, **PIE 0.4.0 native CUDA**, with BF16 activations, weights, and KV cache.

All ten S1 cases used the supplied static prompt, selected examples in retrieval order, and dynamic context. Three repetitions produced **60 measured plans**, plus six discarded warmups. There were **no failures or truncated plans**. No plan-accuracy evaluation was performed.

| Mode | Plans | Decode tokens/s | ms/token | Mean plan decode | Mean generated tokens |
|---|---:|---:|---:|---:|---:|
| Baseline | 30 | 33.20 | 30.12 | 4.29 s | 142.6 |
| Prior-plan speculation | 30 | 40.66 | 24.59 | 3.49 s | 141.9 |

Prior-plan speculation increased decode throughput by **22.5%** and reduced time per token by **18.3%**. It accepted **813 / 1572 proposed draft tokens (51.7%)**, reducing post-bootstrap verifier calls from 4278 to 3444.

## Protocol and limitations

- The dataset has no separate previous-response history. The draft is the first JSON `queue` block in the highest-ranked retrieved task example, a proxy for a previous similar-request plan. The runner records the source example ID/index and exact draft. Current-case generated outputs are never used as drafts. The prior plan is tokenized separately, not appended to the verifier prompt.
- The existing `prior-plan-spec` inferlet uses its original suffix-alignment drafter: match two generated tokens, propose at most eight, and verify with greedy target-model picks. Added `raw_request=true` preserves S1 wording without the software-planning wrapper; 512-token prefill chunks handle the long prompt.
- The normalized decode timer excludes the first sampled token, which is included in prefill/bootstrap. Its token numerator includes the final stop-token step. The original bootstrap-inclusive inferlet metrics are also retained in raw results. Requests are sequential; cache construction/prefill and token-to-text conversion are excluded from the normalized decode comparison.
- Each repetition discards one warmup per mode. Case order rotates and baseline/speculation order alternates. The sampled-token cap is 1,024.
- Exact output text matched in **24/30 paired runs**. The differing cases were `repeat_turn_left` and `photo_of_chair` in each repetition. These runs do **not** establish exact output preservation. Agreement is an implementation diagnostic, not an accuracy score; differing output lengths also affect full-plan latency.
- This standalone test uses selected examples inline. For a controlled masking × prior-plan ablation, use the new combined inferlet's four matched conditions; do not splice this table into the earlier masking table as though prompt layouts and generation paths were identical.

## Reproduce

```bash
cd /workspace/pie-runpod-setup
source env.sh
cargo +1.97.1 build --manifest-path /workspace/pie/inferlets/prior-plan-spec/Cargo.toml --target wasm32-wasip2 --release
python3 scripts/run_s1_prior_plan.py --output results/s1-prior-plan-new-run
```

Start the A3B BF16 native server on port 18085 with `/workspace/pie-runtime/s1-ablation-native-config.toml` before running. Exact inputs, hashes, configuration, raw returns, output plans, and timings are saved with the measurements.

[Raw measurement directory](../../../pie-runpod-setup/results/s1-prior-plan-a3b-bf16-2026-09-13/summary.json) · [Combined inferlet](../../../pie-runpod-setup/inferlets/s1-prior-plan-mask/README.md)
