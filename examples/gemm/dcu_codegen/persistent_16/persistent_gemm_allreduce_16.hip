#ifdef TILELANG_USE_ROCSHMEM
#include <rocshmem/rocshmem.hpp>
#else
#include <dushmem/dushmem.h>
#endif
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "generated_compute_tile.h"

namespace {

#ifdef TILELANG_USE_ROCSHMEM
#define COMM_GLOBAL_EXIT rocshmem::rocshmem_global_exit
#define COMM_PTR rocshmem::rocshmem_ptr
#define COMM_INIT rocshmem::rocshmem_init
#define COMM_MY_PE rocshmem::rocshmem_my_pe
#define COMM_N_PES rocshmem::rocshmem_n_pes
#define COMM_MALLOC rocshmem::rocshmem_malloc
#define COMM_FREE rocshmem::rocshmem_free
#define COMM_BARRIER_ALL rocshmem::rocshmem_barrier_all
#define COMM_FINALIZE rocshmem::rocshmem_finalize
#else
#define COMM_GLOBAL_EXIT dushmem_global_exit
#define COMM_PTR dushmem_ptr
#define COMM_FLOAT_G(address, pe) dushmem_float_g(address, pe)
#define COMM_UINT_P(address, value, pe) dushmem_uint_p(address, value, pe)
#define COMM_INIT dushmem_init
#define COMM_MY_PE dushmem_my_pe
#define COMM_N_PES dushmem_n_pes
#define COMM_MALLOC dushmem_malloc
#define COMM_FREE dushmem_free
#define COMM_BARRIER_ALL dushmem_barrier_all
#define COMM_FINALIZE dushmem_finalize
#endif

constexpr int kM = 4096;
constexpr int kN = 4096;
constexpr int kLocalK = 1024;
constexpr int kTileM = 128;
constexpr int kTileN = 128;
constexpr int kTilesX = kN / kTileN;
constexpr int kTilesY = kM / kTileM;
constexpr int kTileCount = kTilesX * kTilesY;
constexpr int kTileElements = kTileM * kTileN;
constexpr int kDefaultCommBlocks = 16;
constexpr int kThreads = 256;
constexpr int kWarmupIterations = 3;
constexpr int kMeasuredIterations = 10;
// FP32 C_shared is 128 * 128 * sizeof(float) = 64 KiB. The A/B pipeline
// occupies only 32 KiB and aliases this dynamic shared-memory allocation.
constexpr size_t kDynamicSharedBytes = 65536;

#define HIP_CHECK(call)                                                                \
    do {                                                                               \
        hipError_t error = (call);                                                      \
        if (error != hipSuccess) {                                                      \
            std::fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,       \
                         hipGetErrorString(error));                                     \
            COMM_GLOBAL_EXIT(EXIT_FAILURE);                                             \
        }                                                                              \
    } while (0)

__global__ void fill_half(half_t* data, size_t count, float value) {
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        data[index] = static_cast<half_t>(value);
    }
}

extern "C" __global__ __launch_bounds__(kThreads) void persistent_gemm_allreduce_16(
    half_t* __restrict__ A, half_t* __restrict__ B,
    float* __restrict__ partial, float* __restrict__ reduced,
    unsigned int* __restrict__ ready, int my_pe, int npes, int compute_blocks) {
    extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
    const int physical_block = static_cast<int>(blockIdx.x);
    const int comm_blocks = static_cast<int>(gridDim.x) - compute_blocks;

    if (physical_block >= compute_blocks) {
        const int lane = physical_block - compute_blocks;
        for (int tile_id = lane; tile_id < kTileCount; tile_id += comm_blocks) {
            if (threadIdx.x == 0) {
                for (int remote_pe = 0; remote_pe < npes; ++remote_pe) {
                    while (atomicAdd(ready + tile_id * npes + remote_pe, 0U) == 0U) {
                        __builtin_amdgcn_s_sleep(1);
                    }
                }
                __threadfence_system();
            }
            __syncthreads();
            const size_t offset = static_cast<size_t>(tile_id) * kTileElements;
            for (int index = static_cast<int>(threadIdx.x); index < kTileElements;
                 index += kThreads) {
                float sum = 0.0f;
                for (int remote_pe = 0; remote_pe < npes; ++remote_pe) {
                    const float* peer_tile = static_cast<const float*>(
                        COMM_PTR(partial + offset, remote_pe));
#ifdef TILELANG_USE_ROCSHMEM
                    // The IPC backend maps every local peer's symmetric heap into
                    // this process. Host-side validation below guarantees this is
                    // non-null, so no ROC_SHMEM work-group context is needed here.
                    sum += peer_tile[index];
#else
                    sum += peer_tile
                               ? peer_tile[index]
                               : COMM_FLOAT_G(partial + offset + index, remote_pe);
#endif
                }
                reduced[offset + index] = sum;
            }
            __syncthreads();
        }
        return;
    }

    for (int tile_id = physical_block; tile_id < kTileCount; tile_id += compute_blocks) {
        const int logical_bx = tile_id % kTilesX;
        const int logical_by = tile_id / kTilesX;
        const size_t offset = static_cast<size_t>(tile_id) * kTileElements;
        tilelang_compute_tile(A, B, partial + offset, logical_bx, logical_by,
                              buf_dyn_shmem);
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
            for (int target_pe = 0; target_pe < npes; ++target_pe) {
#ifdef TILELANG_USE_ROCSHMEM
                auto* remote_ready = static_cast<unsigned int*>(
                    COMM_PTR(ready + tile_id * npes + my_pe, target_pe));
                atomicExch(remote_ready, 1U);
#else
                COMM_UINT_P(ready + tile_id * npes + my_pe, 1U, target_pe);
#endif
            }
        }
        __syncthreads();
    }
}

}  // namespace

int main(int argc, char** argv) {
    int requested_comm_blocks = kDefaultCommBlocks;
    if (argc == 3 && std::string(argv[1]) == "--comm-blocks") {
        requested_comm_blocks = std::atoi(argv[2]);
    } else if (argc != 1) {
        std::fprintf(stderr, "Usage: %s [--comm-blocks N]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* local_rank_env = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (!local_rank_env) local_rank_env = std::getenv("SLURM_LOCALID");
    const int local_rank = local_rank_env ? std::atoi(local_rank_env) : 0;
    HIP_CHECK(hipInit(0));
    HIP_CHECK(hipSetDevice(local_rank));
    // The Galaxy HIP runtime does not leave a driver context current after
    // hipSetDevice/hipMalloc. DUSHMEM immediately calls hipCtxGetDevice, so
    // explicitly retain and select the device's primary context.
    hipDevice_t hip_device;
    hipCtx_t primary_context;
    HIP_CHECK(hipDeviceGet(&hip_device, local_rank));
    HIP_CHECK(hipDevicePrimaryCtxRetain(&primary_context, hip_device));
    HIP_CHECK(hipCtxSetCurrent(primary_context));
    COMM_INIT();

    int pe = COMM_MY_PE();
    int npes = COMM_N_PES();
    if (npes != 4) {
        if (pe == 0) std::fprintf(stderr, "This prototype requires exactly 4 PEs, got %d\n", npes);
        COMM_GLOBAL_EXIT(EXIT_FAILURE);
    }

    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, local_rank));
    const int total_blocks = properties.multiProcessorCount;
    const int comm_blocks = requested_comm_blocks;
    int compute_blocks = total_blocks - comm_blocks;
    if (comm_blocks <= 0 || compute_blocks <= 0) {
        if (pe == 0) {
            std::fprintf(stderr,
                         "comm_blocks must be in [1, %d], got %d\n",
                         total_blocks - 1, comm_blocks);
        }
        COMM_GLOBAL_EXIT(EXIT_FAILURE);
    }

    hipFuncAttributes kernel_attributes{};
    HIP_CHECK(hipFuncGetAttributes(
        &kernel_attributes,
        reinterpret_cast<const void*>(persistent_gemm_allreduce_16)));
    if (pe == 0) {
        std::fprintf(stderr,
                     "Kernel resources: max_threads=%d registers=%d static_shared=%zu "
                     "local=%zu max_dynamic_shared=%d requested_dynamic_shared=%zu\n",
                     kernel_attributes.maxThreadsPerBlock,
                     kernel_attributes.numRegs,
                     kernel_attributes.sharedSizeBytes,
                     kernel_attributes.localSizeBytes,
                     kernel_attributes.maxDynamicSharedSizeBytes,
                     kDynamicSharedBytes);
    }

    int active_blocks_per_cu = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_cu, persistent_gemm_allreduce_16, kThreads, kDynamicSharedBytes));
    if (active_blocks_per_cu < 1) {
        if (pe == 0) std::fprintf(stderr, "Persistent kernel cannot reside on this device\n");
        COMM_GLOBAL_EXIT(EXIT_FAILURE);
    }

    half_t* A = nullptr;
    half_t* B = nullptr;
    HIP_CHECK(hipMalloc(&A, static_cast<size_t>(kM) * kLocalK * sizeof(half_t)));
    HIP_CHECK(hipMalloc(&B, static_cast<size_t>(kN) * kLocalK * sizeof(half_t)));

    float* partial = static_cast<float*>(
        COMM_MALLOC(static_cast<size_t>(kM) * kN * sizeof(float)));
    float* reduced = static_cast<float*>(
        COMM_MALLOC(static_cast<size_t>(kM) * kN * sizeof(float)));
    unsigned int* ready = static_cast<unsigned int*>(
        COMM_MALLOC(kTileCount * npes * sizeof(unsigned int)));
    if (!partial || !reduced || !ready) {
        std::fprintf(stderr, "PE %d: symmetric allocation failed\n", pe);
        COMM_GLOBAL_EXIT(EXIT_FAILURE);
    }

    HIP_CHECK(hipMemset(partial, 0, static_cast<size_t>(kM) * kN * sizeof(float)));
    HIP_CHECK(hipMemset(reduced, 0, static_cast<size_t>(kM) * kN * sizeof(float)));
    HIP_CHECK(hipMemset(ready, 0, kTileCount * npes * sizeof(unsigned int)));
    constexpr int fill_blocks = 256;
    fill_half<<<fill_blocks, kThreads>>>(A, static_cast<size_t>(kM) * kLocalK, 0.03125f);
    fill_half<<<fill_blocks, kThreads>>>(B, static_cast<size_t>(kN) * kLocalK, 0.0625f);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    COMM_BARRIER_ALL();

#ifdef TILELANG_USE_ROCSHMEM
    // This fused implementation deliberately uses the IPC backend's direct peer
    // mappings. Fail early rather than dereferencing a null device pointer when a
    // non-IPC backend or a topology without peer mappings is selected.
    for (int remote_pe = 0; remote_pe < npes; ++remote_pe) {
        if (!COMM_PTR(partial, remote_pe) || !COMM_PTR(ready, remote_pe)) {
            std::fprintf(stderr,
                         "PE %d: ROC_SHMEM direct mapping to PE %d is unavailable; "
                         "run this prototype with ROCSHMEM_BACKEND=ipc\n",
                         pe, remote_pe);
            COMM_GLOBAL_EXIT(EXIT_FAILURE);
        }
    }
    COMM_BARRIER_ALL();
#endif

    void* args[] = {&A, &B, &partial, &reduced, &ready, &pe, &npes,
                    &compute_blocks};
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    float elapsed_ms = 0.0f;
    double wall_ms = 0.0;
    for (int iteration = -kWarmupIterations; iteration < kMeasuredIterations;
         ++iteration) {
        HIP_CHECK(hipMemset(ready, 0, kTileCount * npes * sizeof(unsigned int)));
        HIP_CHECK(hipDeviceSynchronize());
        COMM_BARRIER_ALL();
        const auto wall_begin = std::chrono::steady_clock::now();
        if (iteration >= 0) HIP_CHECK(hipEventRecord(start));
#ifdef TILELANG_USE_ROCSHMEM
        HIP_CHECK(hipLaunchKernel(
            reinterpret_cast<const void*>(persistent_gemm_allreduce_16),
            dim3(total_blocks), dim3(kThreads), args, kDynamicSharedBytes, nullptr));
#else
        const int launch_status = dushmemx_collective_launch(
            reinterpret_cast<const void*>(persistent_gemm_allreduce_16),
            dim3(total_blocks), dim3(kThreads), args, kDynamicSharedBytes, nullptr);
        if (launch_status != 0) {
            std::fprintf(stderr, "PE %d: dushmemx_collective_launch failed: %d\n",
                         pe, launch_status);
            COMM_GLOBAL_EXIT(EXIT_FAILURE);
        }
#endif
        if (iteration >= 0) {
            HIP_CHECK(hipEventRecord(stop));
            HIP_CHECK(hipEventSynchronize(stop));
            const auto wall_end = std::chrono::steady_clock::now();
            float iteration_ms = 0.0f;
            HIP_CHECK(hipEventElapsedTime(&iteration_ms, start, stop));
            elapsed_ms += iteration_ms;
            wall_ms += std::chrono::duration<double, std::milli>(
                           wall_end - wall_begin).count();
        } else {
            HIP_CHECK(hipDeviceSynchronize());
        }
    }
    elapsed_ms /= kMeasuredIterations;
    wall_ms /= kMeasuredIterations;

    std::vector<float> host_result(static_cast<size_t>(kM) * kN);
    std::vector<float> host_partial(host_result.size());
    HIP_CHECK(hipMemcpy(host_result.data(), reduced,
                        host_result.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(host_partial.data(), partial,
                        host_partial.size() * sizeof(float), hipMemcpyDeviceToHost));
    const float expected = static_cast<float>(npes) * kLocalK * 0.03125f * 0.0625f;
    size_t errors = 0;
    float max_error = 0.0f;
    for (float value : host_result) {
        const float error = std::abs(value - expected);
        max_error = std::max(max_error, error);
        errors += error > 1e-4f;
    }
    float partial_min = host_partial[0];
    float partial_max = host_partial[0];
    size_t partial_errors = 0;
    const float partial_expected = expected / npes;
    for (float value : host_partial) {
        partial_min = std::min(partial_min, value);
        partial_max = std::max(partial_max, value);
        partial_errors += std::abs(value - partial_expected) > 1e-4f;
    }
    std::printf("PE %d: %s, errors=%zu, max_error=%g, event=%.3f ms, "
                "wall=%.3f ms, CUs=%d, "
                "compute_blocks=%d, comm_blocks=%d, partial_errors=%zu, "
                "partial_range=[%g,%g], warmup=%d, iterations=%d\n",
                pe, errors == 0 ? "PASS" : "FAIL", errors, max_error, elapsed_ms,
                wall_ms, total_blocks, compute_blocks, comm_blocks, partial_errors,
                partial_min, partial_max, kWarmupIterations, kMeasuredIterations);
    std::fflush(stdout);

    COMM_BARRIER_ALL();
    COMM_FREE(ready);
    COMM_FREE(reduced);
    COMM_FREE(partial);
    HIP_CHECK(hipFree(B));
    HIP_CHECK(hipFree(A));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));
    COMM_FINALIZE();
    HIP_CHECK(hipDevicePrimaryCtxRelease(hip_device));
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
