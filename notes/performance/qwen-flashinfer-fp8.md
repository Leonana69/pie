# Qwen FP8: native PIE, PIE with FlashInfer, and vLLM

This is an experimental, LLM-assisted native FlashInfer integration on the FP8
branch. It replaces Qwen's routed expert body with FlashInfer's TRT-LLM
block-scaled FP8 MoE implementation. PIE still runs the model, routing, shared
expert, attention, recurrent state, scheduling, and CUDA graphs. The previously
added HC projection/scaled-SiLU fusion remains enabled in both PIE variants.

## Measurement protocol

One NVIDIA B200, single rank, concurrency one, 128 or 1,024 raw input token IDs,
256 output tokens, greedy decoding, ignore EOS, ten requests per prompt length.
Each length has an excluded 256-output warmup. No profiler or sanitizer runs
during timing. Both PIE runs use the same executable and configuration; the
FlashInfer bridge environment variable is the backend selector. Each restart
reloads the checkpoint, because the backends use different weight layouts.
The run order was FlashInfer PIE, native PIE, then vLLM; runs were not
interleaved or randomized, and GPU clocks were not explicitly locked. Small
percentage differences should be interpreted with that limitation.

Both engines use a 128-token forward budget and 2,048-token context limit. Both
place the released n-gram table on the CPU. Requests use fresh KV state; vLLM's
prefix cache is disabled. The synthetic prompt is deliberately reproducible,
but it is not a model-quality evaluation or a maximum-concurrency benchmark.

PIE TTFT is client launch to the first guest sampled-token event; PIE decode is
255 divided by the sum of guest intertoken intervals. vLLM TTFT is HTTP request
to the first nonempty text chunk, and decode is 255 divided by first-to-last
text-chunk time. These transport/observation endpoints differ. The PIE A/B is
more controlled than the cross-engine comparison.

## Results

Medians from ten requests per row; aggregate output throughput includes prefill.

| Input tokens | Engine | TTFT (ms) | Decode (tok/s) | Aggregate output (tok/s) |
|---:|---|---:|---:|---:|
| 128 | PIE native FP8 + fused SiLU | 25.77 | 135.17 | 133.81 |
| 128 | PIE + FlashInfer FP8 MoE | 19.99 | 141.29 | 140.20 |
| 128 | vLLM 0.31.0 | 62.79 | 193.60 | 185.40 |
| 1,024 | PIE native FP8 + fused SiLU | 199.42 | 125.10 | 114.33 |
| 1,024 | PIE + FlashInfer FP8 MoE | 153.66 | 129.86 | 119.77 |
| 1,024 | vLLM 0.31.0 | 459.62 | 193.49 | 140.79 |

FlashInfer improves PIE decode by **4.53% / 3.80%**, and reduces TTFT by
**22.45% / 22.95%**, for 128 / 1,024 input tokens respectively. vLLM still
decodes **37.02% / 49.00% faster** than FlashInfer PIE. Equivalently, FlashInfer
PIE remains 27.02% / 32.89% below vLLM's decode rate.

PIE has lower TTFT in this matched-budget run. This is not a best-possible vLLM
prefill result: both engines are constrained to 128 tokens per forward, and the
vLLM log reports compilation mode NONE with decode CUDA graphs captured at
sizes 1 and 2. Different prefill chunk sizes and execution choices can change
TTFT. These results should not be extrapolated to vLLM with a larger forward
budget. We did not isolate the cause of the TTFT difference with a profile.

Decode ranges were 135.17–135.31, 141.23–141.31, and 193.54–193.69 tok/s for
the short-prompt native PIE, FlashInfer PIE, and vLLM runs. Long-prompt ranges
were 125.08–125.16, 124.38–130.36, and 192.99–193.55 tok/s. FlashInfer PIE had
two slower long-prompt requests; vLLM had a long-prompt TTFT outlier of about
796 ms. Raw measurements are retained, including those requests.

FlashInfer PIE matches native PIE on **19/20 full output sequences** (9/10
short, 10/10 long), and matches vLLM on **20/20**. The native/FlashInfer
mismatch begins at output position 7 (zero-based) in short-prompt request 9.
The decode gain using only the nine matching short sequences is still 4.52%.
All three arithmetic smoke tests produce 391. These simple completion prompts
do not establish broad model-quality parity.

## What changed

- The optional native bridge calls FlashInfer through TVM-FFI. The PIE server
  imports neither Python nor PyTorch. Python is only a preparation/validation tool.
- PIE's existing `MoeExperts` fusion supplies input rows, selected experts, route
  probabilities, and both weight banks to FlashInfer. The shared-expert gate is
  still evaluated by PIE after the routed result.
- The loader retains original FP32 block factors and swaps Qwen's gate/up code
  halves and scales to FlashInfer's expected up/gate order. A layout marker
  prevents ordinary unfused dispatch from consuming the reordered bank.
- Adapter kernels gather the active stage window, quantize BF16 activations in
  128-element groups, transpose scales, zero padded routing weights, and copy
  only active output rows back. Scratch belongs to PIE's stream/region arenas
  and remains valid across graph replay.

The native control uses upstream PIE's aligned Blackwell FP8 gate/up path and
our K=640 down-projection fallback. FlashInfer handles both projections and the
intermediate activation. It accepts K=640 without changing the model dimensions.

## Interpreting the remaining difference

Replacing the routed expert body saves only about 0.29–0.32 ms per decoded token
in this end-to-end A/B, calculated from the reciprocal median decode rates. This
is the net effect of a different MoE implementation, its adapters, weight packing,
and numerical path; it does not measure the original MoE fraction of total time.
It therefore does not establish that MoE is no longer a bottleneck.

There are two concrete areas to investigate next:

1. Match the FlashInfer execution choices. PIE currently uses unshuffled MajorK
   weights and default tactics, whereas vLLM's FP8 expert implementation selects
   shuffled BlockMajorK weights and enters FlashInfer's tuning path. The server
   log also reports monolithic MoE preparation: vLLM passes router
   logits into FlashInfer, while PIE passes already-selected expert IDs and
   weights. PIE also disables PDL inside this body. Packing, tactic choice,
   and launch dependencies
   can affect the same library's performance. Their individual contributions
   have not been isolated here.
2. Profile the work outside that body. PIE still performs routing, shared experts,
   attention/recurrent updates, normalization/projections, n-gram lookup, and its
   own graph/scheduler execution. vLLM reports FlashAttention 4 and a CUDA GDN
   decode kernel. Those paths and PIE's adapter kernels remain distinct even when
   both engines use FlashInfer for routed experts. No new per-kernel profile in
   this experiment assigns a measured share of the gap to any one of them.

These results describe this FP8 port and this synthetic, concurrency-one workload.
They cannot establish a general PIE-versus-vLLM ranking, especially at other
batch sizes or for PIE's original supported model paths.

## Limits of this first integration

This experiment uses FlashInfer 0.7.0.post1, default tactics, unshuffled MajorK
weights, and PDL disabled inside the FlashInfer body. The installed vLLM backend
uses shuffled BlockMajorK weights, FlashInfer autotuning, integrated routing,
and its normal PDL selection. Therefore, calling the same library does not make
the two MoE
execution paths identical. Matching vLLM's packing/tuning is a further experiment,
not a speedup demonstrated by these results.

The implementation is opt-in and currently targets Qwen's resident, single-GPU
FP8 deployment. The native bridge pins a release-specific FlashInfer ABI and
embeds paths to its module/cubin cache. It uses a bounded 192 MiB workspace per
PIE scratch scope and fails on exhaustion. See
[build and activation instructions](../../tools/flashinfer-fp8/README.md).

The vLLM source inspected is version 0.31.0, specifically
`vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py`,
`TrtLlmFp8ExpertsMonolithic._apply_block_scale`: it sets
`use_shuffled_weight=True`, `WeightLayout.BlockMajorK`, supplies `routing_logits`,
and passes `tune_max_num_tokens` to FlashInfer. The actual backend selection and
monolithic preparation are recorded in the retained server log.

The pre-existing recurrent-state precision mismatch also remains: the checkpoint
requests FP32 state, while PIE's Qwen graph declares BF16 state. FlashInfer's MoE
integration does not change that. See the [earlier precision analysis](qwen-fp8-vs-vllm.md#fidelity-and-comparison-limits).
This comparison therefore does not establish equivalent model quality.

## Numerical and validation evidence

The bridge matches FlashInfer's Python API bit-for-bit on random 16-expert
fixtures with hidden size 2,560, intermediate size 640, top-k 10, and 1/4/128
rows, including CUDA graph replay. Relative L2 error against an FP32 arithmetic
reference is approximately 4.6–4.7%, with cosine similarity above 0.9988; this
reference does not emulate FlashInfer's intermediate FP8 quantization.

The PIE GPU test checks non-symmetric gate/up matrices and their scales,
independently generated FlashInfer reference values, nonzero offsets, inactive
rows, empty windows, and graph replay. Its sparse fixture has approximately 6%
relative L2 error against unquantized FP32 arithmetic and up to 13.5% elementwise
relative error. The adapter reproduces FlashInfer's result; passing this check
must not be described as FP32 equivalence.

CUDA memcheck reports zero errors. Racecheck restricted to the added adapter
kernels reports zero hazards. Full racecheck reports hazards inside the supplied
NVIDIA GEMM cubins (86 error and 13 warning groups displayed by Compute Sanitizer
2025.1). These have not been resolved or established as false positives, so the
whole backend does not have a clean racecheck result. Raw logs are retained.

Clippy with warnings denied and the build without CUDA support pass. The complete
model passes the arithmetic smoke prompt. Token parity and final measured values
are recorded with the benchmark evidence below.

## Reproduction and raw evidence

Implementation commit: `c332726d2` on `feat/fp8-activation-support`.
The [evidence directory](flashinfer-fp8-evidence/README.md) contains raw per-request
results, output token IDs, launchers/configuration, binary fingerprints, and
validation logs. The machine-readable [comparison](flashinfer-fp8-evidence/comparison.json)
is generated by [summarize.py](flashinfer-fp8-evidence/summarize.py).

- [Native PIE results](flashinfer-fp8-evidence/native-results.json)
- [PIE FlashInfer results](flashinfer-fp8-evidence/flashinfer-results.json)
- [Fresh vLLM results](flashinfer-fp8-evidence/vllm-results.json)
- [Environment and fingerprints](flashinfer-fp8-evidence/environment.json)
- [Full unresolved racecheck log](flashinfer-fp8-evidence/racecheck.log)

Build/activation and targeted validation commands are in the
[bridge README](../../tools/flashinfer-fp8/README.md). The integration is opt-in;
leaving `PIE_FLASHINFER_MOE_LIBRARY` unset uses the native PIE path.
