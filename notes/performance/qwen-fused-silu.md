# Fused scaled SiLU for the Qwen FP8 deployment

This change adds a fused BF16 hyperconnection projection and scaled SiLU to the
Qwen FP8 branch. Qwen retains BF16 hyperconnection weights even when its experts
use FP8. The new operation computes `BF16(SiLU(scale * BF16(x @ W.T)))` in one
CUDA launch. It does not fuse the routed expert GEMM or change FP8 quantization.

Implementation commit: `fa4718427347cc1da9c71d5903cc7096b42ac82c`.

## Results (2026-10-19)

One B200; concurrency 1; 256 output tokens; medians of ten warmed requests.

| Prompt tokens | Decode before (tok/s) | Decode after (tok/s) | Gain | TTFT before (ms) | TTFT after (ms) |
|---:|---:|---:|---:|---:|---:|
| 128 | 132.76 | 135.22 | +1.85% | 27.89 | 26.63 |
| 1024 | 123.10 | 125.16 | +1.67% | 194.90 | 195.99 |

Aggregate output throughput, including prefill and request overhead, changed
from **131.25 to 133.79 tok/s** at 128 prompt tokens and **112.90 to 114.58 tok/s**
at 1,024. TTFT moved about 1 ms in either direction; this does not establish a
prefill improvement. The prefill path still launches GEMM and SiLU separately.

The isolated projection fell from **4.7510 to 3.5602 microseconds**, a **25.06%**
latency reduction. Across 97 sites that is roughly **0.116 ms per generated
token**. Baseline decode takes 7.53–8.12 ms/token, so a simple sum-of-kernel-times
estimate predicts only about a 1–2% full-model gain. The measured 1.67–1.85% is
consistent with that estimate. Fusion removes launch and intermediate-memory
work; it leaves the much larger expert, attention, and recurrent workloads.
The K=640 FP8 expert down path still uses the port's fallback.

Decode ranges were 132.71–132.87 before versus 135.17–135.38 after for the short
prompt, and 123.07–123.13 versus 125.13–125.19 for the long prompt. This is one
server boot per variant, not a confidence interval over independent restarts.
vLLM was not rerun; the older vLLM comparison is historical context only.

### Output comparison

Both variants answered the arithmetic smoke with `391`. **19 of 20** measured
256-token sequences were identical. Short-prompt request 9 first diverged at
output token index 7; the subsequent autoregressive continuation differed at
238 positions. Recomputing the short-prompt throughput gain using only the nine
matching sequences still gives approximately **1.85%**.

Exact greedy-output parity is therefore **not established**. The changed FP32
reduction tree can change BF16 rounding and an eventual token choice; that is a
plausible explanation, not a demonstrated diagnosis of this request's logits.
The kernel is checked against an FP64 dot-product reference within BF16 output
tolerance, but a model-quality evaluation remains necessary. These raw-token,
EOS-ignored throughput prompts are not a quality benchmark.

Raw evidence: [before](fused-silu-evidence/before-results.json),
[after](fused-silu-evidence/after-results.json),
[comparison](fused-silu-evidence/comparison.json),
[token differences](fused-silu-evidence/token-differences.json),
[kernel timings](fused-silu-evidence/kernel-benchmark-final.log),
[environment](fused-silu-evidence/environment.json), and
[serving configuration](fused-silu-evidence/config.toml).
The resolved [before](fused-silu-evidence/before-profile.toml) and
[after](fused-silu-evidence/after-profile.toml) profiles differ only in the fusion
flag. The [runtime log](fused-silu-evidence/fused-runtime-kernel.log) confirms
that the full model instantiated the new CUDA kernel.

## Implementation

The compiler recognizes `linear.matmul` followed by `elementwise.silu_scaled`
when the projection's intermediate has no outside consumer. The output aliases
the original projection allocation, following the primitive's in-place contract.
CUDA opts into `linear.matmul_silu_scaled`; other backends do not advertise it.

The fast path covers BF16 `[M,10240] @ [320,10240].T`, with one through four
physical/padded rows. This is Qwen's hyperconnection down projection. All other
shapes and encodings run the original matrix multiplication and scaled SiLU.
Prefill therefore retains PIE's GEMM selection. The full 48-layer model has
97 matching sites; the four-layer miniature has nine.

The [kernel](../../crates/kernels-cuda/kernels/linear/silu.cuh) follows PIE's
existing `gemv_splitk_bf16_kernel`: vectorized loads, FP32 accumulation, warp and
shared-memory reductions. It uses 16 warps and two-load unrolling for the long
Qwen reduction, then rounds the dot product to BF16 **before** scaling and SiLU.
This rounding boundary and the order `SiLU(scale * projection)` matter.
Its reduction tree can differ from an autotuned cuBLAS GEMM, so preserving this
boundary does not promise bitwise-identical logits for every input.

The [wrapper](../../crates/kernels-cuda/src/linear/silu.rs) validates shape and
alignment and passes the active-row window. Inactive rows remain untouched.
The launch follows the profile's programmatic dependent launch setting, with
producer waits before reading activations or the stage window.

To disable just this fusion, create
`$PIE_HOME/profiles/cuda/user/nvidia-b200.toml` with:

```toml
api = 1
[target]
device = "NVIDIA B200"
[fuse]
matmul_silu_scaled = false
```

## Boot corrections shared by both measurements

The release stores expert `weight_scale_inv` tensors in **BF16**. The initial
port's importer attempted to reinterpret their bytes as FP32 when stacking
expert legs. Merely constructing the symbolic contract did not catch this;
compiling the real checkpoint's load plan did.

The importer now joins factors in their stored dtype and numerically casts the
whole bank to FP32. Joining before casting also avoids computed-buffer joins
that the CUDA serving loader does not support. Both benchmark variants use
this exact importer correction. Gate/up still uses upstream's UE8M0 scale
repacking; the K=640 expert down fallback retains the widened original factors.

Full boot also exposed an upstream arena reservation bug: the side-stream plan
needed 28,835,840 bytes, but the main plan reserved only 28,397,568. Boot already
tried to commit the larger amount, so it failed against its own smaller ceiling.
It now reserves and commits the larger of the main and side-stream arenas.

Checkpoint loading now keeps at most 32 open shard handles for one load-plan
walk. This avoids reopening NFS files for every small expert extent; the handles
close when that walk finishes. Checkpoint executor tests cover the change.
Neither this I/O optimization nor the arena correction changes model arithmetic.
The measured variants use the same binary, with only the fusion profile flag
changed. Failed boots are excluded from the measurements.

## Measurement protocol

The evidence is collected on one NVIDIA B200, with CUDA runtime/NVRTC 13.0 and
PIE's `cuda_native` backend. The checkpoint is
`Qwen/Qwen3.8-Flash-Next-FP8`, revision
`236dfdf285828023ca3bcd3f37366c58a3469b13`.

Both variants use one request at a time, a 128-token prefill chunk, two state
slots, and a 2,048-token context limit. PIE's current B200 profile keeps the
47.7 GiB n-gram table in pinned host memory. This placement is the same in both
variants. It differs from the earlier experimental build, so the historical
vLLM comparison is not a control for this experiment.

The [benchmark client](fused-silu-evidence/benchmark.py) drives PIE's current
WASI 0.3 `text-completion-bench` inferlet. Each prompt-length case uses ten
requests, identical raw token IDs across variants, greedy sampling, and exactly
256 outputs with EOS ignored. A separate 256-output warmup is excluded. Each
request creates fresh KV state. The arithmetic smoke is checked separately.

- TTFT is measured by the client from launch to arrival of the inferlet's
  first-token message, including host/inferlet overhead.
- Decode throughput is 255 divided by the sum of the 255 guest intertoken
  drain gaps. It is distinct from aggregate output throughput, which includes
  prefill and request overhead.
- Cold checkpoint loading, kernel compilation during warmup, and the smoke
  request are excluded from the reported medians.

Build the current inferlet from the `examples` directory, which selects the
repository's nightly WASI 0.3 toolchain:

```sh
cd examples
cargo build --release --target wasm32-wasip3 -p text-completion-bench
```

Run the script against each already-loaded server (from the repository root):

```sh
python notes/performance/fused-silu-evidence/benchmark.py \
  --checkpoint /path/to/Qwen3.8-Flash-Next-FP8 \
  --wasm /path/to/text_completion_bench.wasm \
  --label before --output /path/to/results
# Repeat with --label after against the fused build.
```

The isolated projection benchmark rotates 97 distinct weight matrices, over
600 MiB in total, rather than repeatedly hitting one matrix in GPU cache. It
captures each sequence as a CUDA graph, warms it, then alternates the before
and after ordering over six repetitions of 100 graph replays. It compares the
fused kernel with PIE's existing GEMM dispatcher plus scaled SiLU, using the
relevant B200 kernel settings.

```sh
cargo test -p kernels-cuda --features cuda --test kernels_cuda \
  benchmark_qwen_scaled_silu_projection -- --ignored --nocapture --test-threads=1
```

## Validation and limits

Tests cover the CPU reference, BF16 projection rounding, negative and zero
scales, one/two/four rows, stage offsets, empty stages, inactive-row preservation,
CUDA graph replay, compiler escape protection, and the 97/9-site full/mini
traces. An engine test compares fusion enabled/disabled in eager and graph modes,
including the prefill fallback. The official-checkpoint test now compiles the
real load plan in addition to checking source mapping. CUDA memcheck and
racecheck, affected-crate Clippy, formatting, and the runtimeless engine build
are also checked.

These are targeted correctness checks, not a model-quality evaluation. The
existing FP8 port's scale-repacking and BF16 recurrent-state limitations remain.
The arithmetic smoke and generated-token comparison are additional evidence,
not proof of general quality parity with vLLM.
