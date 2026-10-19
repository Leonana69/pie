# Experimental native FlashInfer FP8 MoE for Qwen

This LLM-assisted experimental backend calls FlashInfer's TRT-LLM block-FP8 MoE
module from PIE through a small native C ABI. There is no Python interpreter or
PyTorch execution in the PIE server. Python prepares the module and downloads
verified NVIDIA kernel artifacts before serving.

The current integration targets the resident, single-GPU Qwen3.8-Flash-Next FP8
deployment on B200, with up to 128 tokens per forward fire. PIE retains its
router, shared expert, attention, recurrent state, scheduling, and CUDA graphs.
FlashInfer handles the routed expert body and weighted sum. The existing HC
scaled-SiLU fusion can remain enabled.

Build in an environment with a CUDA toolkit, a working PyTorch installation,
`flashinfer-python==0.7.0.post1`, `apache-tvm-ffi==0.1.15`, and `ninja`:

```sh
python tools/flashinfer-fp8/build.py --output /absolute/path/libpie_flashinfer.so
python tools/flashinfer-fp8/prewarm.py
cargo build --release --features cuda
PIE_FLASHINFER_MOE_LIBRARY=/absolute/path/libpie_flashinfer.so target/release/pie -c pie.toml serve
```

The Python environment's `bin` directory must be on `PATH` for `ninja`.
The bridge embeds paths to the built FlashInfer module, its verified cubin cache,
and TVM-FFI. Keep those files in place. It is pinned to that release's native API,
not a stable public FlashInfer ABI. `build.py` defaults to `/usr/local/cuda`; use `--cuda` to select another toolkit. Missing artifacts fail; serving never downloads code.

Unset `PIE_FLASHINFER_MOE_LIBRARY` and restart to use PIE's existing backend.
Keep `fuse.moe_experts` enabled. At load, gate/up halves and their scales are
swapped from PIE's `[gate, up]` order to FlashInfer's `[up, gate]` order, and the
bank is marked so unfused dispatch refuses to misinterpret it. The checkpoint
on disk is unchanged. Original FP32 block factors are retained instead of PIE's
Blackwell scale requantization. Switching back therefore requires a reload.

The bridge uses PIE-owned, stream/region-scoped scratch (192 MiB workspace per
scope). Allocation is bounded and fails on exhaustion. Input preparation gathers
the active stage window, converts BF16 inputs to FP8 under power-of-two 128-wide
scales, and writes padded rows with zero routing weights. Output copying touches
only active rows, preserving inactive storage. FlashInfer PDL is disabled in
this first integration; default tactics are used, with no runtime autotuning.

Numerics differ from the native path: FlashInfer quantizes the first GEMM output
and the activated intermediate; PIE uses different intermediate rounding and,
for aligned banks, different weight-scale packing. Exact token parity and model
quality must be evaluated separately from throughput.

Validate the bridge against FlashInfer's Python API (including graph replay):

```sh
python tools/flashinfer-fp8/validate.py --library /absolute/path/libpie_flashinfer.so --output bridge-check.json
```

To run the optional GPU regression test after prewarming its 16-expert fixture:

```sh
PIE_FLASHINFER_MOE_LIBRARY=/absolute/path/libpie_flashinfer.so \
  cargo test -p kernels-cuda --features cuda the_flashinfer_fp8_body \
  -- --ignored --test-threads=1
```

The test checks sparse non-symmetric gate/up banks, independently generated
FlashInfer reference values, offset and empty stage windows, and graph replay.
