# FlashInfer FP8 comparison evidence

Measured on 2026-10-19 with one NVIDIA B200. See the [analysis](../qwen-flashinfer-fp8.md) for conclusions and limitations.

- `environment.json`: checkpoint revision, implementation commit, measured binary hashes, versions and settings.
- `native-results.json`, `flashinfer-results.json`, `vllm-results.json`: all request timings and output token IDs. The first two also contain guest intertoken intervals.
- `comparison.json`: medians, percentage changes and exact sequence parity, calculated from those raw results.
- `*-smoke.json`: arithmetic checks, separate from performance prompts.
- `*-server.log`, `*-benchmark.log`: actual server and measurement logs.
- `vllm-autotune-configs.json`: eight configurations selected by FlashInfer in the fresh vLLM run.
- `bridge-check-final.json` and log: exact native-bridge versus Python FlashInfer checks, graph replay, and FP32 reference error.
- `kernel-check.log`, `memcheck.log`, `bridge-racecheck.log`: PIE adapter regression test and sanitizer checks.
- `racecheck.log`: **unresolved full-backend hazards**, including supplied NVIDIA GEMM cubins. This is not a passing whole-backend racecheck.
- `sparse-golden.json`, `sparse_reference.py`, `sparse-reference.log`: independent FlashInfer oracle for the sparse PIE fixture and its FP32 error.
- `clippy.log`, `runtimeless-check.log`: Rust static checks.

The PIE client is the unchanged [benchmark.py](../fused-silu-evidence/benchmark.py). The vLLM client is `benchmark_vllm.py`. Each requests ten sequential 256-token completions for each of 128 and 1,024 input tokens. Warmups are excluded. Compare the metric endpoints described in the analysis before interpreting engine differences.

`serve_pie.py`, `serve_vllm.py`, `pie-config.toml`, and `nvidia-b200.toml` retain the exact machine-local paths/settings used. Adapt these paths on another machine. PIE uses the same executable for both variants; set `PIE_FLASHINFER_MOE_LIBRARY` only for the FlashInfer variant. The native library is built with [these tools](../../../tools/flashinfer-fp8/README.md). Checkpoint, executable, native library, and vendor binaries are deliberately not included.

Example PIE client invocation (with the server already ready):

```sh
python notes/performance/fused-silu-evidence/benchmark.py \
  --checkpoint /workspace/models/Qwen3.8-Flash-Next-FP8 \
  --wasm /tmp/pie-bench-wasip3/wasm32-wasip3/release/text_completion_bench.wasm \
  --label native --output /workspace/qwen-flashinfer-benchmark
```

The vLLM client has machine-local model/output paths at its top; update those before running it. Server launchers disable transparent huge pages for that process and place n-gram embeddings on the CPU. No profiler or sanitizer was active during benchmark requests.

The fresh vLLM environment used `vllm==0.31.0`, `nvidia-cuda-nvcc==13.0.88`,
`nvidia-cuda-crt==13.0.88`, `nvidia-nvvm==13.0.88`, and
`nvidia-cuda-cccl==13.0.85`, matching Torch's CUDA 13.0 runtime. Under the pip
CUDA root (`.../site-packages/nvidia/cu13`), `lib64` links to `lib`, and
`lib/libcudart.so` links to `libcudart.so.13`. These packaging adjustments were
needed for FlashInfer JIT compilation and linking. Its separate build cache is
set in the retained launcher. Two startup attempts failed before this repair;
no benchmark samples were collected from them. `vllm-prebuild-fixed.log` records
the successful module build before the measured launch.

Retained text logs normalize line endings and trailing whitespace; message content is unchanged.
