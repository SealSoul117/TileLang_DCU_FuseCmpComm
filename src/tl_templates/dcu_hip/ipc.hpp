#pragma once

// M2 codegen helper.  The descriptor is a device-resident uint64[6] holding
// dcu_ipc::DeviceView.  `trap` is intentional: an invalid runtime peer must
// fail closed rather than turn into an unchecked remote pointer.
#include <dcu_ipc/device.hpp>
#include <dcu_ipc/device_sync.hpp>
#include <cstdint>
#include <cstring>
#include <limits>

namespace tl::ipc {
inline constexpr int kDeviceViewWords = sizeof(dcu_ipc::DeviceView) / sizeof(std::uint64_t);
static_assert(kDeviceViewWords == 6 && alignof(dcu_ipc::DeviceView) <= alignof(std::uint64_t), "descriptor buffer ABI");
// Keep the communication thread coordinate opaque to the TVM and LLVM
// simplifiers.  The tied VGPR operand is intentionally an empty asm block:
// it preserves the exact per-thread value without emitting an extra move.
__device__ __attribute__((always_inline)) inline int opaque_thread_id(int tid) {
  asm volatile("" : "+v"(tid));
  return tid;
}
__device__ __attribute__((always_inline)) inline const dcu_ipc::DeviceView& view(const std::uint64_t* raw) {
  return *reinterpret_cast<const dcu_ipc::DeviceView*>(raw);
}
__device__ __attribute__((always_inline)) inline void check_peer(const dcu_ipc::DeviceView& v, std::int64_t peer, std::int64_t rank, std::int64_t world) {
  if (peer < 0 || peer >= world || world != v.npes || rank != v.my_pe || !v.local_base || !v.peer_bases_device || !v.peer_bases_device[peer]) __builtin_trap();
}
template <class T> __device__ __attribute__((always_inline)) inline T* peer(const std::uint64_t* raw, std::int64_t peer, std::int64_t rank, std::int64_t world, std::size_t bytes = 0) {
  if (!raw) __builtin_trap();
  const auto& v = view(raw); check_peer(v, peer, rank, world);
  if (bytes > v.nbytes || sizeof(T) > v.nbytes - bytes || bytes % alignof(T)) __builtin_trap();
  return dcu_ipc::peer_ptr<T>(v, bytes, static_cast<int>(peer));
}

// A deliberately narrow guard used by the scheduled RS half2 fast path.  It
// performs no payload access and has no synchronization effect.  The compiler
// has already proved the per-lane offsets even; this checks the remaining
// runtime facts which TIR cannot infer from distinct pointer Vars: non-null,
// 4-byte alignment, representable nonempty byte ranges, and that the output
// allocation cannot overlap any read-only peer allocation.  Peer-peer overlap
// is harmless here because all four are only read.
__device__ __attribute__((always_inline)) inline bool can_vectorize_half2(
    const void* p0, std::size_t p0_bytes, const void* p1, std::size_t p1_bytes,
    const void* p2, std::size_t p2_bytes, const void* p3, std::size_t p3_bytes,
    const void* out, std::size_t out_bytes) {
  const auto valid = [](const void* p, std::size_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(p);
    return p != nullptr && bytes >= sizeof(std::uint32_t) && (address & 3U) == 0 &&
           bytes <= std::numeric_limits<std::uintptr_t>::max() - address;
  };
  if (!valid(p0, p0_bytes) || !valid(p1, p1_bytes) || !valid(p2, p2_bytes) ||
      !valid(p3, p3_bytes) || !valid(out, out_bytes)) return false;
  const auto out_address = reinterpret_cast<std::uintptr_t>(out);
  const auto disjoint = [out_address, out_bytes](const void* input, std::size_t input_bytes) {
    const auto input_address = reinterpret_cast<std::uintptr_t>(input);
    // Avoid forming either end address: subtraction is well-defined only on
    // the selected ordering, and touching half-open endpoints is disjoint.
    return out_address <= input_address
               ? input_address - out_address >= out_bytes
               : out_address - input_address >= input_bytes;
  };
  return disjoint(p0, p0_bytes) && disjoint(p1, p1_bytes) &&
         disjoint(p2, p2_bytes) && disjoint(p3, p3_bytes);
}
template <class T> __device__ __attribute__((always_inline)) inline void put_block(T* src, T* dst, std::size_t count) {
  const std::size_t thread = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const std::size_t threads = blockDim.x * blockDim.y * blockDim.z;
  for (std::size_t i = thread; i < count; i += threads) dst[i] = src[i];
}
template <class T> __device__ __attribute__((always_inline)) inline void get_block(const T* src, T* dst, std::size_t count) { put_block(const_cast<T*>(src), dst, count); }
__device__ __attribute__((always_inline)) inline void fence_sys() { dcu_ipc::fence_sys(); }
template <class T> __device__ __attribute__((always_inline)) inline T load_weak(const T* p) { return *p; }
template <class T> __device__ __attribute__((always_inline)) inline void store_weak(T* p, T v) { *p = v; }
template <class T> __device__ __attribute__((always_inline)) inline T load_volatile(const T* p) { return *reinterpret_cast<const volatile T*>(p); }
template <class T> __device__ __attribute__((always_inline)) inline void store_volatile(T* p, T v) { *reinterpret_cast<volatile T*>(p) = v; }
template <class T> __device__ __attribute__((always_inline)) inline T load_relaxed(const T* p, int scope) { return __hip_atomic_load(p, __ATOMIC_RELAXED, scope); }
template <class T> __device__ __attribute__((always_inline)) inline T load_acquire(const T* p, int scope) { return __hip_atomic_load(p, __ATOMIC_ACQUIRE, scope); }
template <class T> __device__ __attribute__((always_inline)) inline void store_relaxed(T* p, T v, int scope) { __hip_atomic_store(p, v, __ATOMIC_RELAXED, scope); }
template <class T> __device__ __attribute__((always_inline)) inline void store_release(T* p, T v, int scope) { __hip_atomic_store(p, v, __ATOMIC_RELEASE, scope); }
}  // namespace tl::ipc
