# S1 RAG — Qwen3.8-27B-FP8: Standalone prior-plan inferlet

2026-09-13, NVIDIA H100 80 GB. **Plain baseline generation works. Prior-plan and exact custom-mask ablations are not validated on this backend.**

## Measured baseline

| Condition | Measured runs | Decode tokens/s | Decode ms/token | Mean decode time/plan | Mean generated tokens |
|---|---:|---:|---:|---:|---:|
| Selected examples inline, no prior plan | 30 | 17.46 | 57.26 | 10.090 s | 176.2 |

All 10 S1 cases ran three times, with three discarded warmups. There were no runtime errors or token-limit truncations in the measured baseline. Mean prompt/bootstrap time was 7.754 s; it is excluded from decode throughput. No plan accuracy was scored.

## Ablation result and limits

A successful inferlet return did not establish correct speculative behavior. In the first-case non-thinking probe, the standalone baseline produced a complete 65-token JSON plan; prior-plan mode stopped after seven malformed tokens (an incomplete JSON fragment). The combined baseline produced 174 tokens; prior-plan-only stopped after nine tokens (`I need to turn around 18 degrees`). Each speculative probe proposed eight draft tokens and accepted none. These are short compatibility probes, not the 30-run performance sample.

The accepted-prefix verifier audit matched both checked anchor positions (one per run), but it did not test post-rejection state restoration. That limited check does not validate the full speculative output. Prior-plan speedup and latency improvement are therefore **not reported as valid comparisons**.

Exact custom masks are not supported by this native model path: `qwen3_5_forward_paged` ignores `mask_d` and `mask_indptr_d`. The 48 recurrent layers retain history as well. Runtime page trimming can still omit whole pages and change outputs; the mask-only and combined probes returning plans is not proof of exact masking. Masking and combined speedups are **N/A**.

## Failure causes

Masking fails because the native hybrid forward does not consume the custom-mask tensors; whole-page trimming does not supply the missing semantics. Prior-plan output failure is reproducible, but its root cause is not isolated: recurrent-state restoration after rejected batched drafts is a suspected cause, and the small anchor audit does not test that restoration.

Earlier execution failures were worked around: the source build fixed missing FP8 scale binding; compatible CUDA components fixed build errors; 64-token prefill avoided an LM-head illegal access consistent with mismatched logits-workspace accounting; skipping upfront CUDA graph capture avoided the first-token crash. The exact overflow and graph pointer defects remain unproven.

See the [failure analysis](../../results/s1-qwen38-fp8-support-2026-09-13/FAILURE_ANALYSIS.md) for raw errors, source evidence, confidence levels, workarounds, and remaining validation.

## Model and protocol

Official [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8), pinned at `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`. E4M3 weights quantized in 128×128 blocks, with BF16 scales; BF16 activations/KV. PIE dequantizes PerGroup FP8 weights to BF16 per GEMM. These are FP8-checkpoint timings, not native FP8-GEMM timings. The model has 48 linear/recurrent layers and 16 full-attention layers.

PIE source `1dd90602`, native CUDA on one H100; greedy sampling; maximum 1024 sampled tokens; 64-token prefill chunks; `PIE_CUDA_DISABLE_UPFRONT_GRAPHS=1`. The official non-thinking prompt format (`enable_thinking=false`) is used uniformly. Dataset content and selected indexes are unchanged. Probe drafts use length 8 and a two-token suffix match, drawn from the first JSON queue in the top-ranked retrieved example as a proxy prior plan.

Decode begins after the first sampled token. Throughput is the sum of subsequent sampled tokens (including stop) divided by summed decode time; generated-token counts exclude stop. Prefill, first-token bootstrap, server load, graph setup before measurement, and shared-catalog construction are separate. No concurrent benchmark ran on the GPU.

## Per-case baseline

| Case | Runs | Decode tokens/s | Decode ms/token | Mean decode time |
|---|---:|---:|---:|---:|
| find_person_shake_hands | 3 | 17.46 | 57.27 | 18.040 s |
| follow_me | 3 | 17.47 | 57.24 | 5.438 s |
| memory_waypoints_visited | 3 | 17.46 | 57.26 | 36.707 s |
| open_greet_calm | 3 | 17.46 | 57.26 | 4.008 s |
| patrol_room | 3 | 17.46 | 57.26 | 6.413 s |
| photo_of_chair | 3 | 17.46 | 57.27 | 6.300 s |
| repeat_turn_left | 3 | 17.46 | 57.28 | 7.904 s |
| strict_turn_bark | 3 | 17.47 | 57.25 | 3.721 s |
| waypoint_sequence | 3 | 17.46 | 57.26 | 7.387 s |
| weather_report | 3 | 17.47 | 57.25 | 4.981 s |

## Reproduction and raw evidence

Measurements, generated text, metadata, configuration, and per-invocation logs: `results/s1-prior-plan-qwen38-fp8-2026-09-13/`. Compatibility probes, memory-checker output, build failures, verified checkpoint hashes, and source snapshots: `results/s1-qwen38-fp8-support-2026-09-13/`. See `QWEN38_FP8_SETUP.md` for the resolved startup/prefill/graph issues and the remaining limitations.

Run `python3 scripts/run_s1_qwen38_fp8_baseline.py --output-root /tmp/qwen38-s1-retest` from the setup repository, with new output directories. It starts/stops the test server, runs only the validated baseline conditions, and uses the same 10-case dataset for both inferlets.
