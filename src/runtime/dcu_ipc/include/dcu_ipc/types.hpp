#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

namespace dcu_ipc {

enum class StatusCode : std::uint8_t {
  Ok,
  InvalidArgument,
  SymmetryMismatch,
  DeviceSelection,
  MultiNodeUnsupported,
  HipError,
  MpiError,
  InvalidState,
  Busy,
};

// Host-only diagnostic. The caller can retain phase/detail after a failed call.
struct Status {
  StatusCode code = StatusCode::Ok;
  std::string phase;
  std::string detail;

  bool ok() const noexcept { return code == StatusCode::Ok; }
  static Status success() { return {}; }
  static Status error(StatusCode code, std::string phase, std::string detail) {
    return {code, std::move(phase), std::move(detail)};
  }
};

enum class MemoryKind : std::uint8_t { DeviceDefault = 0, FineGrained = 1 };
enum class DeviceSelection : std::uint8_t { Auto = 0, ExplicitOrdinal = 1 };

struct InitOptions {
  DeviceSelection selection = DeviceSelection::Auto;
  int ordinal = -1; // Used only with ExplicitOrdinal; ordinal is process-local.
};

// Non-owning GPU argument. All offsets and nbytes are allocation-relative bytes.
// peer_bases_device points to a device-resident array of npes device pointers.
struct DeviceView {
  std::byte* local_base;
  std::byte* const* peer_bases_device;
  std::size_t nbytes;
  int my_pe;
  int npes;
  std::uint64_t allocation_id;
  std::uint64_t generation;
};
static_assert(std::is_standard_layout<DeviceView>::value, "DeviceView must be standard layout");
static_assert(std::is_trivially_copyable<DeviceView>::value, "DeviceView must be trivially copyable");

} // namespace dcu_ipc
