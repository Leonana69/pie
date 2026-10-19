#pragma once
#include "prelude/device.cuh"
#include "prelude/fp8.cuh"

namespace pie::linear {

__device__ __forceinline__ u8 fp8_encode(float x) {
#if __CUDA_ARCH__ >= 890
    unsigned short v;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(v) : "f"(0.f), "f"(x));
    return static_cast<u8>(v);
#else
    return 0;
#endif
}

__device__ __forceinline__ void fp8_mma(float* d, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
#if __CUDA_ARCH__ >= 890
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

template <class T>
__global__ void fp8_quantize_128(const T* x, u8* q, float* scales, int rows, int k,
                               int stage_fan, const u32* win) {
    const int row = blockIdx.x;
    const int group = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (row >= rows) return;
    if (win != nullptr) {
        const int start = stage_fan == 0 ? 0 : static_cast<int>(win[1]) * stage_fan;
        const int count = static_cast<int>(win[0]) * (stage_fan == 0 ? 1 : stage_fan);
        if (row < start || row >= start + count) return;
    }
    const int j = group * 128 + threadIdx.x;
    const float value = j < k ? Elem<T>::to_f32(x[static_cast<long long>(row) * k + j]) : 0.f;
    float amax = fabsf(value);
#pragma unroll
    for (int d = 16; d; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, d));
    __shared__ float maxima[4];
    if (lane == 0) maxima[warp] = amax;
    __syncthreads();
    amax = fmaxf(fmaxf(maxima[0], maxima[1]), fmaxf(maxima[2], maxima[3]));
    const u32 scale_bits = __float_as_uint(fmaxf(amax, 1.e-4f) * (1.f / 448.f));
    const u32 exponent = (scale_bits >> 23) + ((scale_bits & 0x7fffffu) != 0u);
    const float scale = __uint_as_float(exponent << 23);
    if (threadIdx.x == 0) scales[static_cast<long long>(row) * ((k + 127) / 128) + group] = scale;
    if (j < k) q[static_cast<long long>(row) * k + j] = fp8_encode(value / scale);
}

struct Fp8ExpertBases { const u8* codes; const u8* scales; const u8* biases; const u8* pad; };

__device__ __forceinline__ float2 fp8_pair(u32 bits) {
    u32 halfs;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(halfs) : "h"(static_cast<unsigned short>(bits)));
    return __half22float2(*reinterpret_cast<__half2*>(&halfs));
}

template <class T>
__global__ void fp8_expert_stream(const u8* act, const float* as, const i32* routes,
    const u8* codes, const u8* scales, const T* bias, T* out,
    int top_k, int act_div, int n, int k, int experts,
    const Fp8ExpertBases* bases, unsigned int* hits, const u32* win) {
    const int route = blockIdx.x;
    if (win != nullptr && route >= static_cast<int>(win[0]) * top_k) return;
    const int plane = route + (win == nullptr ? 0 : static_cast<int>(win[1]) * top_k);
    const int e = routes[plane];
    const int lane = threadIdx.x & 31;
    const int col = (blockIdx.y * 4 + (threadIdx.x >> 5)) * 4;
    if (col >= n) return;
    if (hits != nullptr && blockIdx.y == 0 && threadIdx.x == 0) atomicAdd(hits, 1u);
    const int blocks = k / 128;
    const int xr = plane / act_div;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    if (e >= 0 && e < experts) {
        const u8* w = (bases == nullptr ? codes : bases->codes) + static_cast<long long>(e) * n * k;
        const float* sf = reinterpret_cast<const float*>(bases == nullptr ? scales : bases->scales)
            + static_cast<long long>(e) * (n / 128) * blocks;
        const uint4* x = reinterpret_cast<const uint4*>(act + static_cast<long long>(xr) * k);
        for (int q = lane; q < k / 16; q += 32) {
            const uint4 av = x[q];
            const float scale = as[static_cast<long long>(xr) * blocks + q / 8] * sf[(col / 128) * blocks + q / 8];
            uint4 weights[4];
#pragma unroll
            for (int r = 0; r < 4; ++r)
                weights[r] = reinterpret_cast<const uint4*>(w + static_cast<long long>(col + r) * k)[q];
            float dot[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const u32 ax = (&av.x)[v];
                const float2 a0 = fp8_pair(ax), a1 = fp8_pair(ax >> 16);
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const u32 wx = (&weights[r].x)[v];
                    const float2 w0 = fp8_pair(wx), w1 = fp8_pair(wx >> 16);
                    dot[r] = fmaf(a1.y, w1.y, fmaf(a1.x, w1.x, fmaf(a0.y, w0.y, fmaf(a0.x, w0.x, dot[r]))));
                }
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) acc[r] = fmaf(dot[r], scale, acc[r]);
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], off);
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < 4; ++r)
            out[static_cast<long long>(plane) * n + col + r] = Elem<T>::from_f32(acc[r]);
    }
}

// Reuse PIE's route-order/work-list construction. Each warp now uses the
// eight MMA columns for eight different routes of the SAME expert, and
// applies the per-128 scale once after four K=32 MMA instructions.
template <class T>
__global__ void fp8_expert_grouped(const u8* act, const float* as,
    const i32* order, const i32* offsets, const i32* work,
    const u8* codes, const u8* scales, T* out,
    int act_div, int n, int k, int experts, int work_cap, int route_count, int top_k,
    const Fp8ExpertBases* bases, unsigned int* hits, const u32* win) {
    const int col = (blockIdx.y * 4 + (threadIdx.x >> 5)) * 16;
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    if (static_cast<int>(blockIdx.x) == work_cap) {
        // The shared sorter puts invalid routes after the valid experts.
        const int live = win == nullptr ? route_count : min(route_count, static_cast<int>(win[0]) * top_k);
        for (int i = offsets[experts]; i < live; ++i)
            if (t == 0) {
                const int plane = order[i];
                out[static_cast<long long>(plane) * n + col + g] = Elem<T>::from_f32(0.f);
                out[static_cast<long long>(plane) * n + col + g + 8] = Elem<T>::from_f32(0.f);
            }
        return;
    }
    if (static_cast<int>(blockIdx.x) >= offsets[experts + 1]) return;
    const int item = work[blockIdx.x], expert = item / 65536, slice = item % 65536;
    const int begin = offsets[expert] + slice * 8;
    const int end = min(offsets[expert + 1], begin + 8);
    if (hits != nullptr && blockIdx.y == 0 && threadIdx.x == 0) atomicAdd(hits, static_cast<unsigned>(end - begin));
    const int xrow = begin + g < end ? order[begin + g] / act_div : -1;
    const int out0 = begin + t * 2 < end ? order[begin + t * 2] : -1;
    const int out1 = begin + t * 2 + 1 < end ? order[begin + t * 2 + 1] : -1;
    const u8* w = (bases == nullptr ? codes : bases->codes) + static_cast<long long>(expert) * n * k;
    const float* sf = reinterpret_cast<const float*>(bases == nullptr ? scales : bases->scales)
        + static_cast<long long>(expert) * (n / 128) * (k / 128);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int b = 0; b < k / 128; ++b) {
        float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int kk = b * 128 + s * 32 + t * 4;
            const u32 a0 = *reinterpret_cast<const u32*>(w + static_cast<long long>(col + g) * k + kk);
            const u32 a1 = *reinterpret_cast<const u32*>(w + static_cast<long long>(col + g + 8) * k + kk);
            const u32 a2 = *reinterpret_cast<const u32*>(w + static_cast<long long>(col + g) * k + kk + 16);
            const u32 a3 = *reinterpret_cast<const u32*>(w + static_cast<long long>(col + g + 8) * k + kk + 16);
            const u32 b0 = xrow < 0 ? 0u : *reinterpret_cast<const u32*>(act + static_cast<long long>(xrow) * k + kk);
            const u32 b1 = xrow < 0 ? 0u : *reinterpret_cast<const u32*>(act + static_cast<long long>(xrow) * k + kk + 16);
            fp8_mma(d, a0, a1, a2, a3, b0, b1);
        }
        const float sw = sf[(col / 128) * (k / 128) + b];
        const float s0 = out0 < 0 ? 0.f : as[static_cast<long long>(out0 / act_div) * (k / 128) + b] * sw;
        const float s1 = out1 < 0 ? 0.f : as[static_cast<long long>(out1 / act_div) * (k / 128) + b] * sw;
        acc[0] = fmaf(d[0], s0, acc[0]); acc[1] = fmaf(d[1], s1, acc[1]);
        acc[2] = fmaf(d[2], s0, acc[2]); acc[3] = fmaf(d[3], s1, acc[3]);
    }
    if (out0 >= 0) {
        out[static_cast<long long>(out0) * n + col + g] = Elem<T>::from_f32(acc[0]);
        out[static_cast<long long>(out0) * n + col + g + 8] = Elem<T>::from_f32(acc[2]);
    }
    if (out1 >= 0) {
        out[static_cast<long long>(out1) * n + col + g] = Elem<T>::from_f32(acc[1]);
        out[static_cast<long long>(out1) * n + col + g + 8] = Elem<T>::from_f32(acc[3]);
    }
}

} // namespace pie::linear
