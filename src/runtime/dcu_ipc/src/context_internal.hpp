#pragma once

#include <dcu_ipc/api.hpp>
#include "metadata.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace dcu_ipc {

struct Allocation {
  std::string tag;
  std::size_t nbytes = 0;
  MemoryKind memory_kind = MemoryKind::DeviceDefault;
  std::uint64_t id = 0;
  std::uint64_t generation = 0;
  void* owner_base = nullptr;
  std::vector<std::byte*> host_peer_bases;
  std::vector<bool> imports_open;
  std::byte** device_peer_bases = nullptr;
  bool live = false;
  unsigned owner_allocated_count = 0;
  unsigned owner_freed_count = 0;
  unsigned opened_imports_count = 0;
  unsigned closed_imports_count = 0;
};

struct Context {
  MPI_Comm comm = MPI_COMM_NULL;
  int rank = -1;
  int npes = 0;
  int local_rank = -1;
  int device_ordinal = -1;
  std::string pci_bdf;
  std::string arch;
  std::uint64_t generation = 0;
  std::uint64_t sequence = 0;
  std::uint64_t next_allocation_id = 1;
  std::size_t live_allocations = 0;
  std::vector<std::unique_ptr<Allocation>> descriptors; // Tombstones survive until finalize.
  std::unordered_set<std::string> live_tags;
  std::thread::id owner_thread;
};

// Explicit restoration avoids HIP calls in a destructor; failures are checked.
class DeviceGuard {
 public:
  Status enter(int selected);
  Status restore();
  void keep_selected() noexcept { previous_ = -1; changed_ = false; }
 private:
  int previous_ = -1;
  bool changed_ = false;
};

Status collective_gate(Context* ctx, metadata::OpKind kind, const Status& local);
Status collective_gate_header(Context* ctx, const metadata::OperationHeader& header,
                              const Status& local);
Context* active_context_ptr() noexcept;
[[noreturn]] void abort_context(MPI_Comm comm, const char* phase, const Status& status);

} // namespace dcu_ipc
