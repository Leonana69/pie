#pragma once
#include "linear/fp8_compute.cuh"
namespace pie::linear {
// Gather the stage window, quantize once, and transpose scales to FlashInfer's
// [K/128, M] layout. Padded rows have zero routing weights and valid expert IDs.
__global__ void flashinfer_fp8_prepare(const bf16* x, const i32* ids, const float* probs,
    u8* q, float* scales, i32* routes, float* weights, int rows, int k, int topk,
    const u32* win) {
    const int row = blockIdx.x, group = blockIdx.y, t = threadIdx.x;
    const int count = win == nullptr ? rows : min(rows, static_cast<int>(win[0]));
    const int source = row + (win == nullptr ? 0 : static_cast<int>(win[1]));
    const float value = row < count ? Elem<bf16>::to_f32(x[static_cast<long long>(source)*k+group*128+t]) : 0.f;
    float amax = fabsf(value);
    for (int d=16; d; d>>=1) amax=fmaxf(amax,__shfl_xor_sync(0xffffffff,amax,d));
    __shared__ float maxima[4];
    if ((t&31)==0) maxima[t>>5]=amax;
    __syncthreads();
    amax=fmaxf(fmaxf(maxima[0],maxima[1]),fmaxf(maxima[2],maxima[3]));
    const u32 bits=__float_as_uint(fmaxf(amax,1.e-4f)/448.f);
    const float scale=__uint_as_float(((bits>>23)+((bits&0x7fffffu)!=0))<<23);
    if (t==0) scales[static_cast<long long>(group)*rows+row]=scale;
    q[static_cast<long long>(row)*k+group*128+t]=fp8_encode(value/scale);
    if (group==0 && t<topk) {
      routes[row*topk+t]=row<count ? ids[source*topk+t] : 0;
      weights[row*topk+t]=row<count ? probs[source*topk+t] : 0.f;
    }
}
__global__ void flashinfer_fp8_finish(const bf16* src, bf16* dst, int rows, int k, const u32* win) {
    const int row=blockIdx.x;
    const int count=win==nullptr ? rows : min(rows,static_cast<int>(win[0]));
    if (row>=count) return;
    const int target=row+(win==nullptr ? 0 : static_cast<int>(win[1]));
    for(int col=threadIdx.x;col<k;col+=blockDim.x)
      dst[static_cast<long long>(target)*k+col]=src[static_cast<long long>(row)*k+col];
}
}
namespace pie::linear {
// FlashInfer's SwiGLU body expects [up, gate]; PIE's checkpoint bank is [gate, up].
__global__ void flashinfer_swap_legs(u32* data, int words, int experts) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=words*experts) return;
  const int e=i/words, j=i%words;
  const u32 a=data[static_cast<long long>(e)*2*words+j];
  data[static_cast<long long>(e)*2*words+j]=data[static_cast<long long>(e)*2*words+words+j];
  data[static_cast<long long>(e)*2*words+words+j]=a;
}
}
