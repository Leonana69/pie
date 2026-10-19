# Qwen FP8 in PIE: implementation and the gap to vLLM

This branch ports the Qwen3.8-Flash-Next-FP8 support onto upstream PIE
`5fcc6f616e26a6a35f3502f90b4fc5ef24d44a7d` (2026-10-19). The measurements below
were collected on **2026-10-18 with an older PIE base**,
`6818175f1047c48a0da487ad41b3aceaa2b184e1`, plus local changes. They are evidence
about that implementation, **not benchmark results for this branch**. Current
upstream has changed enough that it needs a fresh end-to-end comparison.

## What was ported, and what upstream already supplies

Upstream now represents activation precision in the deployment (`a8`) and traced
operations. It already supplies W8A8/W4A8 paths, FP8 checkpoint planes, per-tensor
FP8 embedding gathers, tcgen05 GEMMs, grouped expert execution, and programmatic
dependent launch (PDL). Restoring the old global `fp8_activations` switch would
create a second precision mechanism. This port uses the existing mechanism.

The added deployments are:

- `qwen38-flash-next-bf16-e4m3tile128a8-kv-bf16`
- `qwen38-flash-next-mini-bf16-e4m3tile128a8-kv-bf16`

They retain BF16 dense projections, E4M3 routed experts, and the released E4M3
n-gram table under its scalar factor. Expert activations are quantized in
128-element blocks. This is a mixed-precision model, not an entirely FP8 model.
The full deployment is text-only, single rank, without the MTP draft head.

The Qwen format joins each expert's gate/up codes and their `weight_scale_inv`
factors using upstream's block-FP8 importer. The embedding uses the existing
`e4m3tensor` representation and gather, rather than a new Qwen-only CUDA op.

One backend gap remains relevant: upstream's `lay_tile_banks` requires K to be a
multiple of 512 to pack four scale columns per atom. Qwen's expert down projection
has **K=640**, with five scale columns. Previously it could not reach that W8A8
path. This port supplies a fallback for resident, unlaid E4M3 expert banks:

- [CUDA kernels](../../crates/kernels-cuda/kernels/linear/fp8_compute.cuh): block
  activation quantization, coalesced 16-byte decode loads and FP32 accumulation,
  and route-grouped `mma.sync` prefill.
- [Rust wrapper](../../crates/kernels-cuda/src/linear/fp8_compute.rs): validation,
  stream-scoped scratch, launch selection, and stage-window handling.
- [Engine dispatch](../../crates/engine-cuda/src/dispatch/linear.rs): select the
  fallback when the weight still has its original FP32 block factors.
- [Weight binding](../../crates/engine-cuda/src/weights.rs): retain the E4M3 tag
  so an unlaid FP8 bank cannot be confused with an MXFP4 byte plane.

K and N must be whole 128-element blocks. The fallback quantizes activations
before multiplication; it is not a silent W8A16 fallback. Its narrow decode arm
uses scalar FP32 FMA after reading FP8 operands, while the grouped prefill arm
uses FP8 tensor-core MMA. FP8 operand precision alone does not imply that every
shape uses tensor cores.

The fallback retains the original FP32 weight factors. Aligned banks still use
upstream's Blackwell path, which requantizes factors to power-of-two UE8M0 and
re-rounds codes at load. Thus this branch is **not numerically identical to the
older implementation**, which kept the original factors for both expert legs.
Its Qwen gate/up projection uses the upstream path; its down projection uses the
ported fallback. Full-model quality and performance must be measured again.

The old dense/W4A8 implementations, attention V2 patch, hyperconnection V3 patch,
CUDA 12.8 batch-copy compatibility workaround, and checkpoint-loading diagnostics
are not reapplied. Upstream supplies replacements for several of these. The
fallback kernels themselves require SM90 or newer; the complete deployment's
aligned expert path retains upstream's datacenter Blackwell and NVRTC >=12.9
requirements. The port is validated on B200, not on Hopper or consumer Blackwell.

## Historical experiment

Hardware: one NVIDIA B200, 183359 MiB reported VRAM, driver 595.91.07.
Checkpoint: [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/tree/236dfdf285828023ca3bcd3f37366c58a3469b13),
revision `236dfdf285828023ca3bcd3f37366c58a3469b13`.

Both engines received the same raw synthetic prompt token IDs, temperature zero,
256 output tokens, ignore-EOS enabled, and prefix caching disabled. Each length
had five sequential requests at concurrency one after excluded warmups. CUPTI
collection was disabled during timing. These are latency-oriented single-request
measurements, not maximum server throughput or concurrent-agent benchmarks.

Historical software/configuration:

| Setting | PIE | vLLM |
|---|---|---|
| Software | base `6818175` + local FP8/optimization patches | 0.31.0, commit `db9527a46873454610df6dbedf79a36d6bf1a7f6` |
| CUDA | 12.8, cuBLAS 12.8.4 | PyTorch 2.13.0+cu130, cuBLAS 13.1.1.3 |
| Tensor parallel | 1 | 1 |
| Forward-token budget | 128 | 4096 |
| Model length limit | 2048 | 8192 |
| N-gram placement | GPU | CPU offload |
| Speculation/MTP | disabled | disabled |
| FP8 MoE backend | local stream/grouped kernels | FlashInfer TRT-LLM FP8 MoE |

PIE TTFT ended at the first guest sampled-token event; vLLM TTFT ended at the first
nonempty HTTP text chunk. Decode measurements likewise used guest token events
versus HTTP arrivals. These endpoints differ. In particular, the short-prompt
TTFT result should not be treated as an isolated kernel-latency comparison.

| Historical build | 128-input TTFT (ms) | Decode (tok/s) | 1024-input TTFT (ms) | Decode (tok/s) |
|---|---:|---:|---:|---:|
| Initial local FP8 | 384.14 | 57.85 | 2959.73 | 46.42 |
| V1: optimized FP8 experts only | 81.28 | 85.65 | 577.78 | 61.18 |
| V2: plus selected-attention optimization, fresh rerun | 79.58 | 102.90 | 572.41 | 99.79 |
| V3: plus tiled hyperconnection operations | 74.90 | 113.62 | 557.60 | 109.84 |
| vLLM: fresh comparison | 296.01 | 187.07 | 298.03 | 182.48 |

TTFT and decode are medians. V3's aggregate output throughput including prefill
was 110.35/88.70 tok/s for 128/1024 input tokens; vLLM's was 155.64/145.10.
Fresh vLLM decode varied over 180.74–199.29 and 140.26–197.77 tok/s respectively;
all five requests, including the slowest, were retained. An earlier vLLM run was
193.49/193.21 tok/s. A saved V2 run was 95.98/93.31, versus the fresh
102.90/99.79, despite identical output texts. Use the fresh V2/V3 pair to judge
V3's approximately 10% improvement; the cause of the earlier timing drift was
not isolated. All ten fresh V2/V3 output texts matched, which is a regression
check, not a model-quality evaluation.

[Recorded evidence](fp8-evidence/historical-results.json) includes per-request
metrics, software identifiers, profile summaries, and SHA-256 hashes of the
original result files. The data is historical even when a path or field name
contains “current” or “fresh”.

## Why optimizing FP8 alone did not close the old gap

### Decode: the remaining work was mostly elsewhere

In the V3 decode profile, FP8 expert multiplication took about **0.825 ms/token**,
roughly **10%** of summed kernel time (8.182 ms/token). Activation quantization
added approximately 0.181 ms/token. Optimizing expert GEMMs could still help,
but eliminating 10% of a serial workload gives only about 1.11x speedup under
an idealized Amdahl model; it cannot alone account for the measured 1.65–1.66x
vLLM decode advantage. This is a bound on the profiled component under stated
assumptions, not a prediction of end-to-end speedup.

There were 653 dense GEMV calls per token in old PIE: 290 hyperconnection
projections, 192 router/shared-expert/gate calls, 108 GDN calls, 60 attention
and indexer calls, two PLE projections, and the vocabulary head. Those calls
reflect the model graph rather than an accidental extra forward pass.

Isolated GEMV probes using the same PyTorch/CUDA 13 stack as vLLM showed most
PIE shapes competitive; the 10240x320 hyperconnection up projection favored
Torch. “Much of the time is in dense GEMV” does not establish that replacing
all GEMVs with cuBLAS would be faster.

### Model-level fusion matters

[vLLM's hyperconnection implementation](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/models/qwen4_exp/nvidia/hyperconnection.py)
merges the down and injection projections, fuses the SiLU epilogue, and delays
residual combination to fuse it with the next normalization. PIE's Qwen model
still expresses these as separate operations in
[forward.poem](../../models/qwen_4/forward.poem). Compiler fusion must be checked
in the executed plan before estimating any remaining savings.

The old one-CTA-per-token HC mix/inject kernels underused the GPU; tiling them
produced the measured V3 gain. Current upstream already has vectorized,
spread-out HC implementations. Reapplying the old tiling patch or claiming that
current PIE still launches one CTA would be incorrect.

[vLLM's QSA implementation](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/models/qwen4_exp/nvidia/qsa.py)
also merges QKV/gate projection and fuses normalization/rotary/gate handling.
The PIE model expresses Q/gate, K and V projections separately. Their actual
launch costs require a new profile on this branch.

### Scheduling: historical overlap is evidence, not a speedup estimate

| Instrumented decode | Old PIE V3 | vLLM |
|---|---:|---:|
| Kernel records per token | 1886 | 1713 |
| Summed kernel durations per token | 8.182 ms | 7.575 ms |
| Union of kernel intervals per token | 8.181 ms | 5.100 ms |

PIE's old custom launches were effectively serial. vLLM uses PDL and a
[prefetching HC down-projection variant](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/models/qwen4_exp/nvidia/ops/cute_dsl/hc_down_silu.py).
However, intervals can overlap while kernels wait for dependencies. The
2.475 ms difference between vLLM's sum and union is not measured useful work
or guaranteed recoverable time. No isolated PDL ablation was performed.

Current upstream PIE **already supports PDL**, including the launch attribute
and kernel synchronization. The old conclusion “PIE has no PDL” no longer
applies. The ported fallback does not enable PDL: enabling it safely would
require a matching in-kernel wait/trigger protocol, followed by race testing
and an end-to-end ablation.

### Prefill: FP8 multiplication and selected attention were both expensive

Separate profiles of one 1024-input/one-output request measured:

| Kernel category | Old PIE V3 | vLLM | PIE/vLLM |
|---|---:|---:|---:|
| Expert matrix multiplication | 163.465 ms | 14.980 ms | 10.91x |
| Selected attention | 153.913 ms | 2.029 ms | 75.87x |
| Union of all GPU kernel intervals | 497.835 ms | 38.405 ms | 12.96x |

These are **instrumented kernel times**, not end-to-end latency ratios. PIE
processed eight 128-token chunks; vLLM processed one 1024-token chunk. The
chunk-size effect was not isolated. CUPTI perturbed both engines, particularly
vLLM decode; timeline gaps cannot be interpreted directly as production CPU
cost. The measured 1024-input TTFT ratio was about 1.87x, not 13x.

The ported grouped fallback reuses weights across eight routes per expert and
uses four K=32 MMA instructions before applying a 128-element scale. It remains
a relatively simple implementation: separate quantization and routing passes,
small tiles, and no Blackwell TMA/tcgen05 pipeline. vLLM's recorded TRT-LLM MoE
kernels use Blackwell-specific FP8 matrix multiplication, scale handling and
fusion. This is a plausible disadvantage for the fallback during prefill;
its contribution in the new mixed-backend deployment has not been measured.

The old selected-attention prefill loop accounted for another 31% of recorded
PIE kernel time. Changing FP8 cannot remove that cost. Current upstream has
also changed attention kernels, so that old 75.87x ratio needs remeasurement.
A dense-attention fast path is valid only when selection provenance and bounds
prove that all relevant keys are selected; a large top-k setting alone is
insufficient for arbitrary sparse indices.

## Fidelity and comparison limits

The checkpoint specifies `text_config.mamba_ssm_dtype = "float32"`.
[vLLM's model configuration](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/models/config.py)
honors this setting. PIE's Qwen model still declares BF16 recurrent delta state
in [forward.poem](../../models/qwen_4/forward.poem). This pre-existing mismatch
is unresolved. It does not explain slower PIE (BF16 state is cheaper), but it
prevents a claim of equivalent model quality. Neither arithmetic smoke tests
nor numerical kernel tests establish language-model quality.

Other limits are the differing CUDA libraries, prefill budgets, n-gram placement,
streaming endpoints, synthetic prompt distribution, single GPU and concurrency
one. The old results cannot establish general superiority of either engine or
contradict claims made for different models and agent/KV workloads.

## Next experiments

1. Benchmark this exact branch against the pinned vLLM build, recording commit,
   compiler/library versions, output counts, timing endpoints and raw samples.
   Separate fresh performance results from the historical table above.
2. Profile which expert legs use tcgen05, the new fallback, and any fused path.
   Compare K=640 fallback against an explicitly padded/compatible tcgen05 path;
   preserve logical scale indexing and evaluate the extra memory and rounding.
3. Vary the prefill token budget with a measured memory budget. More tokens per
   pass can improve route reuse, but the 185.5 GB checkpoint leaves limited room
   on one B200; fitting a larger batch must be demonstrated.
4. Inspect the executed Qwen HC/attention graph and fuse remaining projection,
   SiLU, residual, normalization, and quantization boundaries where profitable.
   Preserve intended BF16 rounding boundaries or quantify numerical changes.
5. Evaluate PDL/prefetch for the fallback with a controlled on/off experiment.
   Report uninstrumented latency separately from timeline overlap.
6. Implement FP32 recurrent state and run a model-quality comparison, including
   the effects of upstream's scale requantization, before claiming parity.

## Validation of this port

See [FP8 support instructions](../fp8-support.md) for commands and the validation
record. These checks establish import/dispatch and kernel correctness; no new
full-model throughput result is claimed for this branch.
