// Opt-in host adapter for an exported, nine-argument GEMM+ReduceScatter HIP
// kernel.  This is intentionally separate from TileLang's generated ctypes
// adapter: its only job is to make an external source artifact measurable by
// the existing benchmark harness without representing it as a recompile.
#include <hip/hip_runtime.h>

#ifndef DIAGNOSTIC_HIP_SOURCE
#error "DIAGNOSTIC_HIP_SOURCE must name the exported HIP source"
#endif
#ifndef DIAGNOSTIC_HIP_KERNEL
#error "DIAGNOSTIC_HIP_KERNEL must name the exported HIP kernel symbol"
#endif
#ifndef DIAGNOSTIC_HIP_GRID_X
#error "DIAGNOSTIC_HIP_GRID_X must provide the fixed diagnostic grid"
#endif

#include DIAGNOSTIC_HIP_SOURCE

extern "C" int gemm_rs_diagnostic_hip_launch(
    void* a, void* b, void* ipc_view_partial, void* ipc_view_ready, void* out,
    void* remote_partial3, void* remote_ready3, int rank, int world_size,
    void* stream) {
  hipLaunchKernelGGL(
      DIAGNOSTIC_HIP_KERNEL, dim3(DIAGNOSTIC_HIP_GRID_X), dim3(512), 64 * 1024,
      reinterpret_cast<hipStream_t>(stream), reinterpret_cast<half_t*>(a),
      reinterpret_cast<half_t*>(b), reinterpret_cast<uint64_t*>(ipc_view_partial),
      reinterpret_cast<uint64_t*>(ipc_view_ready), reinterpret_cast<half_t*>(out),
      reinterpret_cast<half_t*>(remote_partial3), reinterpret_cast<uint*>(remote_ready3),
      rank, world_size);
  return static_cast<int>(hipGetLastError());
}
