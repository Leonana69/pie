#pragma once
#include "prelude/device.cuh"

namespace pie::linear {

// PIE's gemv_splitk reduction with the scaled-SiLU epilogue. The selected
// 16-warp geometry targets Qwen HC; preserve the BF16 rounding boundary.
template <int kWarps, int kUnrollP = 1>
__global__ void matmul_silu_scaled_bf16_kernel(
    const bf16* __restrict__ weight,
    const bf16* __restrict__ act,
    bf16* __restrict__ out,
    int N, int K, float scale, const u32* win)
{
    pdl_wait();
    pdl_launch();
    const int token = blockIdx.y;
    if (win != nullptr && token >= win[0]) return;
    const int at = token + (win != nullptr ? win[1] : 0);
    act += static_cast<long long>(at) * K;
    out += static_cast<long long>(at) * N;
    const int row = blockIdx.x;
    if (row >= N) return;
    const int warp = threadIdx.y;
    const float4* w4 =
        reinterpret_cast<const float4*>(weight + (long long)row * K);
    const float4* x4 = reinterpret_cast<const float4*>(act);
    const int vectors = K / 8;

    float acc = 0.f;

    constexpr int kU = kUnrollP;
    const int stride = kWarps * 32;
    int i = warp * 32 + threadIdx.x;
    for (; i + stride * (kU - 1) < vectors; i += stride * kU) {
        float4 wv[kU];
        float4 xv[kU];
        #pragma unroll
        for (int u = 0; u < kU; ++u) {
            wv[u] = w4[i + stride * u];
            xv[u] = x4[i + stride * u];
        }
        #pragma unroll
        for (int u = 0; u < kU; ++u) {
            const bf16* wb = reinterpret_cast<const bf16*>(&wv[u]);
            const bf16* xb = reinterpret_cast<const bf16*>(&xv[u]);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc += bf16_to_f32(wb[j]) * bf16_to_f32(xb[j]);
            }
        }
    }
    for (; i < vectors; i += stride) {
        float4 wv = w4[i];
        float4 xv = x4[i];
        const bf16* wb = reinterpret_cast<const bf16*>(&wv);
        const bf16* xb = reinterpret_cast<const bf16*>(&xv);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc += bf16_to_f32(wb[j]) * bf16_to_f32(xb[j]);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }

    __shared__ float partial[kWarps];
    if (threadIdx.x == 0) partial[warp] = acc;
    __syncthreads();
    if (warp != 0 || threadIdx.x != 0) return;
    float total = 0.f;
    #pragma unroll
    for (int w = 0; w < kWarps; ++w) total += partial[w];
    // Preserve the unfused BF16 projection rounding BEFORE scale and SiLU.
    const float v = bf16_to_f32(f32_to_bf16(total)) * scale;
    out[row] = f32_to_bf16(v / (1.f + __expf(-v)));
}

} // namespace pie::linear
