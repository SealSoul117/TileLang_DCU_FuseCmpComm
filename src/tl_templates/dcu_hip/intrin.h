#pragma once

#include "common.h"

namespace tl {

TL_DEVICE int get_warp_idx_sync(int warp_size, int thread_id) {
  int warp_idx = thread_id / warp_size;
  return __builtin_amdgcn_readfirstlane(warp_idx);
}

TL_DEVICE int get_warp_idx_sync(int warp_size = 64) {
  return get_warp_idx_sync(warp_size, static_cast<int>(threadIdx.x));
}

TL_DEVICE int get_warp_idx(int warp_size = 64) {
  return static_cast<int>(threadIdx.x) / warp_size;
}

TL_DEVICE int get_lane_idx(int warp_size = 64) {
  // mbcnt reports the hardware's linear lane ID.  It agrees with the
  // historical threadIdx.x % 64 expression only for the 1-D, wave-aligned
  // blocks used by the DCU GEMM kernels.  Preserve the latter for every
  // other shape/size so this helper does not change unrelated 2-D mappings.
  if (warp_size == 64 && warpSize == 64 && (blockDim.x % 64) == 0) {
    unsigned int lo = __builtin_amdgcn_mbcnt_lo(~0u, 0);
    return __builtin_amdgcn_mbcnt_hi(~0u, lo);
  }
  return static_cast<int>(threadIdx.x) % warp_size;
}

}  // namespace tl
