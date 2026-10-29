#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace dcu_ipc {

// Thread-level primitives for naturally aligned uint32_t signals in live,
// IPC-mapped memory whose fine-grained system atomics have been verified on the
// target GPUs. Payload memory must also belong to a verified memory-mode
// combination (this project's test covers fine-grained flags AND payload).
// An acquire makes payload reads valid only after observing the corresponding
// release in a correct ownership protocol. Each slot has one writer; the
// application owns its epoch/ACK protocol. uint32_t wrap is NOT handled;
// equality waits may miss an overwritten value, and >= is not wrap-safe.
// These calls do not publish a block's other threads, make a stale DeviceView
// safe, or guarantee that an unresponsive peer will make progress. In
// particular, wait may never return. No host/MPI operation is performed.
__device__ __attribute__((always_inline)) inline void fence_sys() {
  __threadfence_system();
}

__device__ __attribute__((always_inline)) inline void
signal_set_release(std::uint32_t* signal, std::uint32_t value) {
  __hip_atomic_store(signal, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __attribute__((always_inline)) inline std::uint32_t
signal_load_acquire(const std::uint32_t* signal) {
  return __hip_atomic_load(signal, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __attribute__((always_inline)) inline bool
signal_test_eq_acquire(const std::uint32_t* signal, std::uint32_t value) {
  return signal_load_acquire(signal) == value;
}

__device__ __attribute__((always_inline)) inline bool
signal_test_ge_acquire(const std::uint32_t* signal, std::uint32_t value) {
  return signal_load_acquire(signal) >= value;
}

__device__ __attribute__((always_inline)) inline void
wait_eq_acquire(const std::uint32_t* signal, std::uint32_t value) {
  while (!signal_test_eq_acquire(signal, value)) {
    // Power/issue backoff only: the acquire load above is the synchronization.
    __builtin_amdgcn_s_sleep(1);
  }
}

__device__ __attribute__((always_inline)) inline void
wait_ge_acquire(const std::uint32_t* signal, std::uint32_t value) {
  while (!signal_test_ge_acquire(signal, value)) {
    __builtin_amdgcn_s_sleep(1);
  }
}

} // namespace dcu_ipc
