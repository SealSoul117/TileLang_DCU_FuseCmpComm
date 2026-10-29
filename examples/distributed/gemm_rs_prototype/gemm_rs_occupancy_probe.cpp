// Tiny host-only helper for the bounded scheduled GEMM+RS runner.  Keeping
// hipDeviceProp_t in C++ avoids guessing its ABI or HIP enum values in ctypes.
#include <hip/hip_runtime.h>

extern "C" int gemm_rs_scheduled_cu_count(int* count) {
  if (count == nullptr) return static_cast<int>(hipErrorInvalidValue);
  int device = -1;
  hipError_t status = hipGetDevice(&device);
  if (status != hipSuccess) return static_cast<int>(status);
  hipDeviceProp_t properties{};
  status = hipGetDeviceProperties(&properties, device);
  if (status != hipSuccess) return static_cast<int>(status);
  *count = properties.multiProcessorCount;
  return static_cast<int>(hipSuccess);
}
