#pragma once
#define HIP_ENABLE_WARP_SYNC_BUILTINS
#include <hip/hip_runtime.h>
#include <tl_templates/dcu_hip/gemm.h>
#include <tl_templates/dcu_hip/copy.h>
#include <tl_templates/dcu_hip/reduce.h>
#include <tl_templates/dcu_hip/ldsm.h>
#include <tl_templates/dcu_hip/threadblock_swizzle.h>
#include <tl_templates/dcu_hip/debug.h>


__device__ __forceinline__ void tilelang_compute_tile(
    half_t* __restrict__ A, half_t* __restrict__ B,
    float* __restrict__ C_tile, int logical_bx, int logical_by,
    uchar* buf_dyn_shmem) {
  float* C = C_tile;

  float C_local[64];
  half_t A_local[16];
  half_t B_local[16];
  for (int i = 0; i < 16; ++i) {
    *(float4*)(C_local + (i * 4)) = make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    *(uint4*)(((half_t*)buf_dyn_shmem) + ((((((int64_t)i_1) * (int64_t)2048) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)3) ^ ((((int64_t)((int)threadIdx.x)) & (int64_t)15) >> (int64_t)2)) * (int64_t)8)) + (int64_t)8192)) = *(uint4*)(A + (((((logical_by) * 131072) + (i_1 * 65536)) + ((((int)threadIdx.x) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 8)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    *(uint4*)(((half_t*)buf_dyn_shmem) + (((((int64_t)i_2) * (int64_t)2048) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)3) ^ ((((int64_t)((int)threadIdx.x)) & (int64_t)15) >> (int64_t)2)) * (int64_t)8))) = *(uint4*)(B + (((((logical_bx) * 131072) + (i_2 * 65536)) + ((((int)threadIdx.x) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 8)));
  }
  for (int ko = 0; ko < 31; ++ko) {
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      *(uint4*)(((half_t*)buf_dyn_shmem) + (((((((((int64_t)ko) + (int64_t)1) & (int64_t)1) * (int64_t)4096) + (((int64_t)i_3) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)3) ^ ((((int64_t)((int)threadIdx.x)) & (int64_t)15) >> (int64_t)2)) * (int64_t)8)) + (int64_t)8192)) = *(uint4*)(A + (((((((logical_by) * 131072) + (i_3 * 65536)) + ((((int)threadIdx.x) >> 2) * 1024)) + (ko * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 32));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      *(uint4*)(((half_t*)buf_dyn_shmem) + ((((((((int64_t)ko) + (int64_t)1) & (int64_t)1) * (int64_t)4096) + (((int64_t)i_4) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)3) ^ ((((int64_t)((int)threadIdx.x)) & (int64_t)15) >> (int64_t)2)) * (int64_t)8))) = *(uint4*)(B + (((((((logical_bx) * 131072) + (i_4 * 65536)) + ((((int)threadIdx.x) >> 2) * 1024)) + (ko * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 32));
    }
    __syncthreads();
    for (int ki = 0; ki < 2; ++ki) {
      for (int i_5 = 0; i_5 < 4; ++i_5) {
        *(uint2*)(A_local + (i_5 * 4)) = *(uint2*)(((half_t*)buf_dyn_shmem) + ((((((((((int64_t)ko) & (int64_t)1) * (int64_t)4096) + (((((int64_t)((int)threadIdx.x)) & (int64_t)127) >> (int64_t)6) * (int64_t)2048)) + (((int64_t)i_5) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)32)) + ((((((int64_t)ki) * (int64_t)2) + ((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5)) ^ (((int64_t)((int)threadIdx.x)) & (int64_t)3)) * (int64_t)8)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)4) * (int64_t)4)) + (int64_t)8192));
      }
      for (int j = 0; j < 4; ++j) {
        *(uint2*)(B_local + (j * 4)) = *(uint2*)(((half_t*)buf_dyn_shmem) + ((((((((((int64_t)ko) & (int64_t)1) * (int64_t)4096) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)7) * (int64_t)2048)) + (((int64_t)j) * (int64_t)512)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) & (int64_t)3) * (int64_t)128)) + ((((((int64_t)((int)threadIdx.x)) & (int64_t)63) & (int64_t)15) >> (int64_t)2) * (int64_t)32)) + ((((((int64_t)ki) * (int64_t)2) + ((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5)) ^ (((((int64_t)((int)threadIdx.x)) & (int64_t)63) & (int64_t)15) >> (int64_t)2)) * (int64_t)8)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)4) * (int64_t)4)));
      }
      for (int i_6 = 0; i_6 < 4; ++i_6) {
        for (int j_1 = 0; j_1 < 4; ++j_1) {
          {
    *(((float32x4*)C_local) + ((i_6 * 4) + j_1)) = __builtin_amdgcn_mmac_f32_16x16x16f16(*(((float16x4*)A_local) + j_1),
                  *(((float16x4*)B_local) + i_6),
                  *(((float32x4*)C_local) + ((i_6 * 4) + j_1)));
  };
        }
      }
    }
  }
  for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      *(uint2*)(A_local + (i_7 * 4)) = *(uint2*)(((half_t*)buf_dyn_shmem) + ((((((((((int64_t)((int)threadIdx.x)) & (int64_t)127) >> (int64_t)6) * (int64_t)2048) + (((int64_t)i_7) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)32)) + ((((((int64_t)ki_1) * (int64_t)2) + ((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5)) ^ (((int64_t)((int)threadIdx.x)) & (int64_t)3)) * (int64_t)8)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)4) * (int64_t)4)) + (int64_t)12288));
    }
    for (int j_2 = 0; j_2 < 4; ++j_2) {
      *(uint2*)(B_local + (j_2 * 4)) = *(uint2*)(((half_t*)buf_dyn_shmem) + ((((((((((int64_t)((int)threadIdx.x)) >> (int64_t)7) * (int64_t)2048) + (((int64_t)j_2) * (int64_t)512)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) & (int64_t)3) * (int64_t)128)) + ((((((int64_t)((int)threadIdx.x)) & (int64_t)63) & (int64_t)15) >> (int64_t)2) * (int64_t)32)) + ((((((int64_t)ki_1) * (int64_t)2) + ((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5)) ^ (((((int64_t)((int)threadIdx.x)) & (int64_t)63) & (int64_t)15) >> (int64_t)2)) * (int64_t)8)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)4) * (int64_t)4)) + (int64_t)4096));
    }
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      for (int j_3 = 0; j_3 < 4; ++j_3) {
        {
    *(((float32x4*)C_local) + ((i_8 * 4) + j_3)) = __builtin_amdgcn_mmac_f32_16x16x16f16(*(((float16x4*)A_local) + j_3),
                  *(((float16x4*)B_local) + i_8),
                  *(((float32x4*)C_local) + ((i_8 * 4) + j_3)));
  };
      }
    }
  }
  __syncthreads();
  for (int i_9 = 0; i_9 < 4; ++i_9) {
    for (int j_4 = 0; j_4 < 4; ++j_4) {
      *(float4*)(((float*)buf_dyn_shmem) + (((((((((int)threadIdx.x) >> 7) * 8192) + (j_4 * 2048)) + (((((int)threadIdx.x) & 127) >> 6) * 1024)) + (i_9 * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((int)threadIdx.x) & 63) >> 4) * 4))) = *(float4*)(C_local + ((j_4 * 16) + (i_9 * 4)));
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_10 = 0; i_10 < 16; ++i_10) {
    *(float4*)(C + (((i_10 * 1024) + ((((int)threadIdx.x) >> 5) * 128)) + ((((int)threadIdx.x) & 31) * 4))) = *(float4*)(((float*)buf_dyn_shmem) + ((((((((int)threadIdx.x) & 31) >> 2) * 2048) + (i_10 * 128)) + ((((int)threadIdx.x) >> 5) * 16)) + ((((int)threadIdx.x) & 3) * 4)));
  }
}
