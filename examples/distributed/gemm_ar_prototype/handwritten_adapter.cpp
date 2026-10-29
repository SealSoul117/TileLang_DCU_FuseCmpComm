// Comparison only: include the unchanged, explicitly selected handwritten v2
// source. This file is never used to implement the TileLang kernel.
#ifndef GEMM_AR_REFERENCE_SOURCE
#error "Define GEMM_AR_REFERENCE_SOURCE to an existing handwritten v2 .hip"
#endif
#define main gemm_ar_reference_unused_main
#include GEMM_AR_REFERENCE_SOURCE
#undef main

static_assert(sizeof(dcu_ipc::DeviceView) == 6 * sizeof(uint64_t));

extern "C" int gemm_ar_reference_shape(int* m, int* n, int* k) {
  *m = FUSED_M; *n = FUSED_N; *k = FUSED_K;
  return 0;
}

extern "C" const void* gemm_ar_reference_kernel() {
  return reinterpret_cast<const void*>(fused_vector);
}

// Descriptors point to HOST copies of the ordinary six-word DeviceView.
// Snapshotting them is done outside timing by the caller; kernels receive
// their original by-value DeviceView ABI, with no descriptor-copy kernel.
extern "C" int gemm_ar_reference_launch(
    void* a, void* b, const uint64_t* stage_words,
    const uint64_t* out_words, const uint64_t* ready_words,
    int rank, int total_blocks, int compute_blocks, void* stream) {
  dcu_ipc::DeviceView stage, output, ready;
  std::memcpy(&stage, stage_words, sizeof(stage));
  std::memcpy(&output, out_words, sizeof(output));
  std::memcpy(&ready, ready_words, sizeof(ready));
  hipLaunchKernelGGL(fused_vector, dim3(total_blocks), dim3(Threads),
                    SharedBytes, reinterpret_cast<hipStream_t>(stream),
                    static_cast<half_t*>(a), static_cast<half_t*>(b),
                    stage, output, ready, rank, compute_blocks);
  return static_cast<int>(hipGetLastError());
}
