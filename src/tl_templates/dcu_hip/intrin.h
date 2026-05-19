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
  return static_cast<int>(threadIdx.x) % warp_size;
}

}  // namespace tl
