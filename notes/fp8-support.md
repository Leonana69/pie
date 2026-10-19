# Qwen FP8 support

This is an experimental, LLM-assisted port onto PIE upstream
`5fcc6f616e26a6a35f3502f90b4fc5ef24d44a7d`. It adds the Qwen FP8 deployment and
checkpoint mapping, plus an isolated W8A8 expert fallback for banks that cannot
use upstream's Blackwell scale packing. The existing deployment precision API
selects FP8 activations; the earlier experimental `fp8_activations` configuration
option is not used on this branch.

See [the implementation and performance analysis](performance/qwen-fp8-vs-vllm.md)
for benchmark provenance, remaining bottlenecks, and numerical limitations.
The [scaled-SiLU fusion report](performance/qwen-fused-silu.md) covers the new
fused hyperconnection projection and the fresh before/after experiment.

## Build and import

Use the repository's pinned Rust toolchain and a datacenter Blackwell GPU with
NVRTC 12.9 or newer for the complete Qwen deployment. The port's validation used
one B200 and NVRTC/CUDA runtime 13.0 for engine tests. Ensure the intended CUDA
libraries are actually loaded when multiple versions are installed.

```sh
cargo build -p pie --release --features cuda --bin pie
pie model import /path/to/Qwen3.8-Flash-Next-FP8 \
  --deployment qwen38-flash-next-bf16-e4m3tile128a8-kv-bf16 \
  --keep-source
```

Add `--dry-run` to inspect the import plan before writing an artifact. Serve the
imported deployment through PIE's CUDA backend. Keep ordinary activations BF16;
expert activation quantization is encoded by the deployment's `a8` precision.
Do not reuse the older benchmark's `fp8_activations = true` setting.

The full deployment retains BF16 dense weights, FP8 experts, and the FP8 n-gram
table with its stored scalar factor. It is single rank, text-only, without MTP.
The corresponding miniature deployment is
`qwen38-flash-next-mini-bf16-e4m3tile128a8-kv-bf16`. Kernel-level FP16 input support
does not imply a separate FP16 full-model deployment.

The fallback does not introduce expert offload or support arbitrary quantization
formats. It accepts whole 128-element weight blocks and resident expert banks
without affine biases. Aligned banks continue through upstream's backend;
Qwen's K=640 down projection keeps its original block factors widened to FP32 and uses the
fallback. The two paths have different weight-scale rounding, so this port
requires a fresh model-quality comparison.

## Validation

Run GPU tests serially, including the CUDA graph tests:

```sh
cargo test -p kernels-cuda --features cuda --test kernels_cuda \
  the_fp8_experts_preserve_block_factors -- --test-threads=1
cargo test -p engine-cuda --features cuda --test engine_cuda \
  an_fp8_expert_with_five_scale_columns -- --test-threads=1
cargo test -p pie-poem-compiler --test poem_compiler \
  the_qwen4_fp8 -- --test-threads=1
```

The optional official-checkpoint test reads metadata from the safetensors
shards, constructs the full import contract, and checks expert/embedding source
mapping, and compiles the load plan against the real metadata. This checks
that BF16 expert factors are numerically widened before use as FP32 scales.
It does not load all model tensors onto the GPU:

```sh
PIE_QWEN_FP8_CHECKPOINT=/path/to/Qwen3.8-Flash-Next-FP8 \
  cargo test -p pie-poem-compiler --test poem_compiler \
  the_qwen4_fp8 -- --include-ignored --test-threads=1
```

For memory and race checks, run the built `kernels_cuda` test executable under
`compute-sanitizer --tool memcheck --error-exitcode 99` and
`compute-sanitizer --tool racecheck --error-exitcode 99`, passing
`the_fp8_experts_preserve_block_factors --test-threads=1` to the executable.

Validation recorded on 2026-10-19:

- Three CUDA reference/graph tests: decode and grouped prefill, per-token and
  per-route activations, K=640, non-power-of-two weight factors, invalid routes,
  stage offsets, inactive output preservation, and FP16 graph replay.
- Engine integration: load a tiled FP8 artifact with five scale columns, then
  execute prefill and decode in eager and graph modes against expected logits.
- Full/mini deployment traces and the official checkpoint's import contract.
- CUDA memcheck: zero errors; racecheck: zero hazards for the fallback tests.
- Formatting, affected-crate Clippy with warnings denied, and CUDA engine build
  checking.

A fresh full-model fusion A/B is recorded in the
[scaled-SiLU report](performance/qwen-fused-silu.md): 1.67–1.85% faster decode
on B200, with 19/20 measured output sequences matching. This is not a general
model-quality evaluation. The historical vLLM comparison in the older analysis
still comes from an older PIE implementation. The checkpoint's requested FP32
recurrent state is still BF16 in PIE's Qwen model, as detailed in the analysis.
