#include <dushmem/dushmem.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "generated_compute_tile.h"

namespace {

constexpr int kM = 4096;
constexpr int kN = 4096;
constexpr int kLocalK = 1024;
constexpr int kTileM = 128;
constexpr int kTileN = 128;
constexpr int kTilesX = kN / kTileN;
constexpr int kTilesY = kM / kTileM;
constexpr int kTileCount = kTilesX * kTilesY;
constexpr int kTileElements = kTileM * kTileN;
constexpr int kThreads = 256;
constexpr int kWarmupIterations = 3;
constexpr int kMeasuredIterations = 10;
constexpr size_t kDynamicSharedBytes = 65536;

#define HIP_CHECK(call)                                                            \
    do {                                                                           \
        hipError_t error = (call);                                                  \
        if (error != hipSuccess) {                                                  \
            std::fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,  \
                         hipGetErrorString(error));                                 \
            dushmem_global_exit(EXIT_FAILURE);                                      \
        }                                                                          \
    } while (0)

__global__ void fill_half(half_t* data, size_t count, float value) {
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        data[index] = static_cast<half_t>(value);
    }
}

extern "C" __global__ __launch_bounds__(kThreads) void tilelang_gemm_all_tiles(
    half_t* __restrict__ A, half_t* __restrict__ B,
    float* __restrict__ partial) {
    extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
    const int tile_id = static_cast<int>(blockIdx.x);
    const int logical_bx = tile_id % kTilesX;
    const int logical_by = tile_id / kTilesX;
    tilelang_compute_tile(A, B, partial + static_cast<size_t>(tile_id) * kTileElements,
                          logical_bx, logical_by, buf_dyn_shmem);
}

}  // namespace

int main() {
    const int local_rank = std::getenv("SLURM_LOCALID")
                               ? std::atoi(std::getenv("SLURM_LOCALID"))
                               : 0;
    HIP_CHECK(hipInit(0));
    HIP_CHECK(hipSetDevice(local_rank));
    hipDevice_t hip_device;
    hipCtx_t primary_context;
    HIP_CHECK(hipDeviceGet(&hip_device, local_rank));
    HIP_CHECK(hipDevicePrimaryCtxRetain(&primary_context, hip_device));
    HIP_CHECK(hipCtxSetCurrent(primary_context));
    dushmem_init();

    int pe = dushmem_my_pe();
    const int npes = dushmem_n_pes();
    if (npes != 4) {
        if (pe == 0) std::fprintf(stderr, "Expected four PEs, got %d\n", npes);
        dushmem_global_exit(EXIT_FAILURE);
    }

    half_t* A = nullptr;
    half_t* B = nullptr;
    HIP_CHECK(hipMalloc(&A, static_cast<size_t>(kM) * kLocalK * sizeof(half_t)));
    HIP_CHECK(hipMalloc(&B, static_cast<size_t>(kN) * kLocalK * sizeof(half_t)));
    float* partial = static_cast<float*>(
        dushmem_malloc(static_cast<size_t>(kM) * kN * sizeof(float)));
    float* reduced = static_cast<float*>(
        dushmem_malloc(static_cast<size_t>(kM) * kN * sizeof(float)));
    if (!partial || !reduced) {
        std::fprintf(stderr, "PE %d: symmetric allocation failed\n", pe);
        dushmem_global_exit(EXIT_FAILURE);
    }

    fill_half<<<256, kThreads>>>(A, static_cast<size_t>(kM) * kLocalK, 0.03125f);
    fill_half<<<256, kThreads>>>(B, static_cast<size_t>(kN) * kLocalK, 0.0625f);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    dushmem_barrier_all();

    hipEvent_t begin, gemm_done, reduce_done;
    HIP_CHECK(hipEventCreate(&begin));
    HIP_CHECK(hipEventCreate(&gemm_done));
    HIP_CHECK(hipEventCreate(&reduce_done));
    float gemm_ms = 0.0f;
    float reduce_ms = 0.0f;
    double wall_ms = 0.0;

    for (int iteration = -kWarmupIterations; iteration < kMeasuredIterations;
         ++iteration) {
        HIP_CHECK(hipMemset(reduced, 0,
                            static_cast<size_t>(kM) * kN * sizeof(float)));
        HIP_CHECK(hipDeviceSynchronize());
        dushmem_barrier_all();
        const auto wall_begin = std::chrono::steady_clock::now();
        HIP_CHECK(hipEventRecord(begin));
        tilelang_gemm_all_tiles<<<kTileCount, kThreads, kDynamicSharedBytes>>>(
            A, B, partial);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipEventRecord(gemm_done));
        const int reduce_status = dushmemx_float_sum_reduce_on_stream(
            DUSHMEM_TEAM_WORLD, reduced, partial,
            static_cast<size_t>(kM) * kN, nullptr);
        if (reduce_status != 0) {
            std::fprintf(stderr, "PE %d: DUSHMEM reduce returned %d\n", pe,
                         reduce_status);
            dushmem_global_exit(EXIT_FAILURE);
        }
        HIP_CHECK(hipEventRecord(reduce_done));
        HIP_CHECK(hipEventSynchronize(reduce_done));
        const auto wall_end = std::chrono::steady_clock::now();
        if (iteration >= 0) {
            float one_gemm_ms = 0.0f;
            float one_reduce_ms = 0.0f;
            HIP_CHECK(hipEventElapsedTime(&one_gemm_ms, begin, gemm_done));
            HIP_CHECK(hipEventElapsedTime(&one_reduce_ms, gemm_done, reduce_done));
            gemm_ms += one_gemm_ms;
            reduce_ms += one_reduce_ms;
            wall_ms += std::chrono::duration<double, std::milli>(
                           wall_end - wall_begin).count();
        }
    }
    gemm_ms /= kMeasuredIterations;
    reduce_ms /= kMeasuredIterations;
    wall_ms /= kMeasuredIterations;

    std::vector<float> host_result(static_cast<size_t>(kM) * kN);
    HIP_CHECK(hipMemcpy(host_result.data(), reduced,
                        host_result.size() * sizeof(float), hipMemcpyDeviceToHost));
    const float expected = npes * kLocalK * 0.03125f * 0.0625f;
    size_t errors = 0;
    float max_error = 0.0f;
    for (float value : host_result) {
        const float error = std::abs(value - expected);
        max_error = std::max(max_error, error);
        errors += error > 1e-4f;
    }
    std::printf("PE %d: %s, GEMM=%.3f ms, DUSHMEM_AllReduce=%.3f ms, "
                "event_total=%.3f ms, wall_total=%.3f ms, errors=%zu, max_error=%g, "
                "warmup=%d, iterations=%d\n",
                pe, errors == 0 ? "PASS" : "FAIL", gemm_ms, reduce_ms,
                gemm_ms + reduce_ms, wall_ms, errors, max_error, kWarmupIterations,
                kMeasuredIterations);
    std::fflush(stdout);

    dushmem_barrier_all();
    dushmem_free(reduced);
    dushmem_free(partial);
    HIP_CHECK(hipFree(B));
    HIP_CHECK(hipFree(A));
    HIP_CHECK(hipEventDestroy(reduce_done));
    HIP_CHECK(hipEventDestroy(gemm_done));
    HIP_CHECK(hipEventDestroy(begin));
    dushmem_finalize();
    HIP_CHECK(hipDevicePrimaryCtxRelease(hip_device));
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
