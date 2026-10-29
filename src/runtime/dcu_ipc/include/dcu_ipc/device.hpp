#pragma once

#include <dcu_ipc/types.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace dcu_ipc {

enum class DevicePtrError : std::uint8_t {
  Ok = 0, EmptyView, InvalidPeer, OutOfRange, Unaligned, InvalidLocalPointer,
};
template <class T> struct DevicePtrResult {
  T* pointer;
  DevicePtrError error;
};

// HIP device-only header: CPU code must use remote_ptr, since the peer table
// itself lives in device memory. T must be a complete object type.
// Fast path preconditions: live view; valid peer; byte_offset begins a complete,
// naturally aligned T within this allocation. The caller establishes data
// readiness and view lifetime via host collectives. A freed or corrupted view
// cannot be made safe by the checked path. No implicit fence/atomic.
template <class T>
__device__ __attribute__((always_inline)) inline T* peer_ptr(DeviceView view,
                                                 std::size_t byte_offset, int peer) {
  return reinterpret_cast<T*>(view.peer_bases_device[peer] + byte_offset);
}

// local_pointer must point within this view's owning allocation. The offset
// is computed in bytes; no device-side scan of allocation descriptors occurs.
template <class T>
__device__ __attribute__((always_inline)) inline T* ptr(DeviceView view, T* local_pointer, int peer) {
  const auto offset = reinterpret_cast<std::uintptr_t>(local_pointer) -
                      reinterpret_cast<std::uintptr_t>(view.local_base);
  return peer_ptr<T>(view, offset, peer);
}

template <class T>
__device__ __attribute__((always_inline)) inline DevicePtrResult<T>
checked_peer_ptr(DeviceView view, std::size_t byte_offset, int peer) {
  if (!view.local_base || !view.peer_bases_device || view.nbytes == 0 || view.npes < 1)
    return {nullptr, DevicePtrError::EmptyView};
  if (peer < 0 || peer >= view.npes) return {nullptr, DevicePtrError::InvalidPeer};
  if (byte_offset > view.nbytes || sizeof(T) > view.nbytes - byte_offset)
    return {nullptr, DevicePtrError::OutOfRange};
  auto* base = view.peer_bases_device[peer];
  if (!base) return {nullptr, DevicePtrError::EmptyView};
  const auto address = reinterpret_cast<std::uintptr_t>(base);
  if (address > std::numeric_limits<std::uintptr_t>::max() - byte_offset)
    return {nullptr, DevicePtrError::OutOfRange};
  if ((address + byte_offset) % alignof(T)) return {nullptr, DevicePtrError::Unaligned};
  return {reinterpret_cast<T*>(base + byte_offset), DevicePtrError::Ok};
}

template <class T>
__device__ __attribute__((always_inline)) inline DevicePtrResult<T>
checked_ptr(DeviceView view, T* local_pointer, int peer) {
  if (!view.local_base || !view.peer_bases_device || view.nbytes == 0 || view.npes < 1)
    return {nullptr, DevicePtrError::EmptyView};
  const auto base = reinterpret_cast<std::uintptr_t>(view.local_base);
  const auto address = reinterpret_cast<std::uintptr_t>(local_pointer);
  if (!local_pointer || address < base || address - base > view.nbytes)
    return {nullptr, DevicePtrError::InvalidLocalPointer};
  return checked_peer_ptr<T>(view, address - base, peer);
}

} // namespace dcu_ipc
