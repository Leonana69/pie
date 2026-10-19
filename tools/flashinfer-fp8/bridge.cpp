// Optional native bridge to FlashInfer 0.7.0.post1's TRT-LLM FP8 MoE module.
// Python is used only by build.py. Serving calls the TVM-FFI module directly.
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/container/tensor.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <memory>
#include <iterator>
#include <stdexcept>

namespace ffi = tvm::ffi;
namespace {
thread_local std::string last_error;
thread_local char* arena;
thread_local size_t capacity, cursor;
struct Allocation {
  DLManagedTensorVersioned managed{};
  std::vector<int64_t> shape, strides;
};
int allocate(DLTensor* prototype, DLManagedTensorVersioned** out, void* error_ctx,
             void (*set_error)(void*, const char*, const char*)) noexcept {
  try {
    if (prototype->device.device_type != kDLCUDA) throw std::runtime_error("non-CUDA workspace requested");
    auto a = std::make_unique<Allocation>();
    a->shape.assign(prototype->shape, prototype->shape + prototype->ndim);
    a->strides.resize(prototype->ndim);
    size_t count = 1;
    for (int i = prototype->ndim - 1; i >= 0; --i) {
      a->strides[i] = count;
      if (a->shape[i] < 0 || (a->shape[i] && count > SIZE_MAX / a->shape[i]))
        throw std::runtime_error("workspace shape overflow");
      count *= a->shape[i];
    }
    const size_t item = (prototype->dtype.bits * prototype->dtype.lanes + 7) / 8;
    if (item && count > SIZE_MAX / item) throw std::runtime_error("workspace byte overflow");
    size_t bytes = count * item;
    size_t start = (cursor + 255) & ~size_t(255);
    if (start > capacity || bytes > capacity - start)
      throw std::runtime_error("FlashInfer workspace exhausted: need at least " + std::to_string(start + bytes));
    a->managed.version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    a->managed.manager_ctx = a.get();
    a->managed.deleter = [](DLManagedTensorVersioned* p) { delete static_cast<Allocation*>(p->manager_ctx); };
    a->managed.dl_tensor = *prototype;
    a->managed.dl_tensor.data = arena + start;
    a->managed.dl_tensor.shape = a->shape.data();
    a->managed.dl_tensor.strides = a->strides.data();
    a->managed.dl_tensor.byte_offset = 0;
    cursor = start + bytes;
    *out = &a.release()->managed;
    return 0;
  } catch (const std::exception& e) { set_error(error_ctx, "RuntimeError", e.what()); return -1; }
}
using SetCubin = void (*)(const char*, int);
SetCubin set_cubin;
void cubin(const char* name, const char*) {
  // build.py downloads and verifies artifacts before serving. No network here.
  std::ifstream f(std::string(PIE_FI_CUBINS) + "/" + name, std::ios::binary);
  if (!f) { last_error = std::string("Missing prewarmed FlashInfer cubin: ") + name; set_cubin("", 0); return; }
  std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  set_cubin(bytes.data(), static_cast<int>(bytes.size()));
}
ffi::Function& function() {
  static ffi::Module module = [] {
    // FlashInfer JIT modules resolve TVM-FFI through the process symbol scope.
    if (!dlopen("libtvm_ffi.so", RTLD_NOW | RTLD_GLOBAL)) throw std::runtime_error(dlerror());
    void* handle = dlopen(PIE_FI_MODULE, RTLD_NOW | RTLD_LOCAL);
    if (!handle) throw std::runtime_error(dlerror());
    set_cubin = reinterpret_cast<SetCubin>(dlsym(handle, "FlashInferSetCurrentCubin"));
    auto setter = reinterpret_cast<void (*)(void (*)(const char*, const char*))>(dlsym(handle, "FlashInferSetCubinCallback"));
    if (!set_cubin || !setter) throw std::runtime_error("FlashInfer cubin ABI missing");
    setter(cubin);
    return ffi::Module::LoadFromFile(PIE_FI_MODULE);
  }();
  static ffi::Function fn = module->GetFunction("trtllm_fp8_block_scale_moe").value();
  return fn;
}
struct View {
  std::vector<int64_t> shape, strides;
  DLTensor tensor{};
  View(void* ptr, std::initializer_list<int64_t> dims, DLDataType dtype, int device)
      : shape(dims), strides(shape.size()) {
    int64_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) { strides[i] = stride; stride *= shape[i]; }
    tensor = {ptr, {kDLCUDA, device}, static_cast<int>(shape.size()), dtype, shape.data(), strides.data(), 0};
  }
  ffi::TensorView get() const { return ffi::TensorView(&tensor); }
};
struct Context {
  DLPackManagedTensorAllocator previous_alloc;
  void* previous_stream;
  int device;
  Context(void* stream, int d) : device(d) {
    TVMFFIEnvSetDLPackManagedTensorAllocator(allocate, 0, &previous_alloc);
    TVMFFIEnvSetStream(kDLCUDA, d, stream, &previous_stream);
  }
  ~Context() {
    TVMFFIEnvSetStream(kDLCUDA, device, previous_stream, nullptr);
    TVMFFIEnvSetDLPackManagedTensorAllocator(previous_alloc, 0, nullptr);
  }
};
}
extern "C" const char* pie_flashinfer_error() { return last_error.c_str(); }
extern "C" int pie_flashinfer_abi() { return 1; }
extern "C" int pie_flashinfer_fp8(
    void* stream, void* workspace, size_t workspace_bytes,
    void* x, void* xs, void* ids, void* probabilities,
    void* w1, void* s1, void* w2, void* s2, void* out,
    int rows, int hidden, int intermediate, int experts, int topk) noexcept {
  try {
    last_error.clear();
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) throw std::runtime_error("cudaGetDevice failed");
    arena = static_cast<char*>(workspace); capacity = workspace_bytes; cursor = 0;
    Context context(stream, device);
    const DLDataType fp8{kDLFloat8_e4m3fn, 8, 1}, f32{kDLFloat, 32, 1}, bf16{kDLBfloat,16,1}, i32{kDLInt,32,1};
    View a(x,{rows,hidden},fp8,device), as(xs,{hidden/128,rows},f32,device);
    View routes(ids,{rows,topk},i32,device), weights(probabilities,{rows,topk},f32,device);
    View up(w1,{experts,2*intermediate,hidden},fp8,device), us(s1,{experts,2*intermediate/128,hidden/128},f32,device);
    View down(w2,{experts,hidden,intermediate},fp8,device), ds(s2,{experts,hidden/128,intermediate/128},f32,device);
    View y(out,{rows,hidden},bf16,device);
    auto result = function()(
      int64_t(2), nullptr, routes.get(), weights.get(), nullptr, a.get(), as.get(),
      up.get(), us.get(), nullptr, nullptr, nullptr, nullptr, nullptr,
      down.get(), ds.get(), nullptr, y.get(),
      int64_t(experts), int64_t(topk), int64_t(0), nullptr, nullptr,
      int64_t(intermediate), int64_t(0), int64_t(experts), 1.0,
      int64_t(4), false, int64_t(0), true, false, ffi::Array<int64_t>{-1,-1},
      int64_t(1), int64_t(3), true, nullptr, ffi::Array<ffi::Tensor>{}, ffi::Array<ffi::Tensor>{},
      false, nullptr, nullptr);
    if (!last_error.empty()) throw std::runtime_error(last_error);
    return 0;
  } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
