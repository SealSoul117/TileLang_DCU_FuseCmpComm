#include "context_internal.hpp"
#include "bootstrap_mpi.hpp"

#include <hip/hip_runtime.h>

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <thread>
#include <vector>

namespace dcu_ipc {
namespace {
Status bad(StatusCode code, const char* phase, const char* detail) {
  return Status::error(code, phase, detail);
}
Status hip_result(hipError_t rc, const char* phase) {
  if (rc == hipSuccess) return Status::success();
  return Status::error(StatusCode::HipError, phase,
                       std::to_string(int(rc)) + " " + hipGetErrorString(rc));
}
Status agree(Context* ctx, const Status& local, const char* phase) {
  return bootstrap::collective_status(ctx->comm, local, phase);
}
void fatal_if_error(Context* ctx, const Status& status, const char* phase) {
  if (!status.ok()) abort_context(ctx->comm, phase, status);
}
void log_resource(Context* ctx, const char* phase, const Allocation* a) {
  std::printf("RESOURCE rank=%d phase=%s id=%llu tag=%s owner_alloc=%u owner_free=%u imports_opened=%u imports_closed=%u status=PASS\n",
              ctx->rank, phase, static_cast<unsigned long long>(a->id), a->tag.c_str(),
              a->owner_allocated_count, a->owner_freed_count,
              a->opened_imports_count, a->closed_imports_count);
  std::fflush(stdout);
}
bool injected(const char* stage) {
#ifdef DCU_IPC_TEST_FAULTS
  const char* requested = std::getenv("DCU_IPC_TEST_FAIL");
  return requested && std::string(requested) == stage;
#else
  (void)stage;
  return false;
#endif
}
Status injection(const char* stage, Context* ctx) {
  std::fprintf(stderr, "INJECT rank=%d stage=%s\n", ctx->rank, stage);
  std::fflush(stderr);
  return bad(StatusCode::HipError, stage, "test-only injected failure");
}
Allocation* find_descriptor(Context* ctx, const Allocation* pointer) {
  if (!ctx || !pointer) return nullptr;
  for (auto& owned : ctx->descriptors)
    if (owned.get() == pointer) return owned.get(); // No dereference of unknown pointer.
  return nullptr;
}
Status restore_or_abort(Context* ctx, DeviceGuard& guard) {
  auto status = guard.restore();
  fatal_if_error(ctx, status, "restore caller device");
  return status;
}
void close_imports(Context* ctx, Allocation* a) {
  for (int peer = 0; peer < ctx->npes; ++peer) if (a->imports_open[peer]) {
    fatal_if_error(ctx, hip_result(hipIpcCloseMemHandle(a->host_peer_bases[peer]),
                                   "hipIpcCloseMemHandle"), "close import");
    a->imports_open[peer] = false;
    a->host_peer_bases[peer] = nullptr;
    ++a->closed_imports_count;
  }
  log_resource(ctx, "local-imports-closed", a);
  fatal_if_error(ctx, agree(ctx, Status::success(), "all imports closed"), "close agreement");
  log_resource(ctx, "all-imports-closed", a);
}
void free_owned_resources(Context* ctx, Allocation* a) {
  if (a->device_peer_bases) {
    fatal_if_error(ctx, hip_result(hipFree(a->device_peer_bases), "hipFree pointer table"),
                   "free pointer table");
    a->device_peer_bases = nullptr;
  }
  if (a->owner_base) {
    fatal_if_error(ctx, hip_result(hipFree(a->owner_base), "hipFree owner"), "free owner");
    a->owner_base = nullptr;
    ++a->owner_freed_count;
  }
  log_resource(ctx, "owner-and-table-freed", a);
  fatal_if_error(ctx, agree(ctx, Status::success(), "all owners freed"), "free agreement");
}
void rollback(Context* ctx, Allocation* a, DeviceGuard& guard) {
  fatal_if_error(ctx, hip_result(hipDeviceSynchronize(), "rollback HIP completion"),
                 "rollback synchronize");
  fatal_if_error(ctx, agree(ctx, Status::success(), "rollback completed"), "rollback sync agreement");
  close_imports(ctx, a);
  free_owned_resources(ctx, a);
  restore_or_abort(ctx, guard);
  log_resource(ctx, "rollback-complete", a);
}
}

Status symm_alloc(Context* ctx, const char* tag, std::size_t nbytes,
                  MemoryKind kind, Allocation** out) {
  Context* active = active_context_ptr();
  if (!active) return bad(StatusCode::InvalidState, "alloc", "no active Context");
  if (std::this_thread::get_id() != active->owner_thread)
    return bad(StatusCode::InvalidState, "alloc", "wrong thread; no MPI collective entered");
  Status local = ctx == active ? Status::success() :
                 bad(StatusCode::InvalidState, "alloc", "null or foreign Context");
  if (local.ok()) local = metadata::validate_tag(tag, {});
  if (local.ok()) for (const auto& live : active->live_tags)
    if (live == tag) { local = bad(StatusCode::InvalidState, "alloc", "tag already live"); break; }
  if (local.ok() && nbytes == 0) local = bad(StatusCode::InvalidArgument, "alloc", "zero bytes");
  if (local.ok() && !out) local = bad(StatusCode::InvalidArgument, "alloc", "null output");
  if (local.ok() && *out) local = bad(StatusCode::InvalidState, "alloc", "output already populated");
  if (local.ok() && active->next_allocation_id == std::numeric_limits<std::uint64_t>::max())
    local = bad(StatusCode::InvalidState, "alloc", "allocation id exhausted");
  metadata::OperationHeader header{};
  auto built = metadata::make_header(metadata::OpKind::Alloc, active->sequence,
                                     active->next_allocation_id, active->generation,
                                     tag, nbytes, kind, &header);
  if (local.ok() && !built.ok()) local = built;
  auto status = collective_gate_header(ctx, header, local);
  if (!status.ok()) return status;

  std::unique_ptr<Allocation> pending(new (std::nothrow) Allocation);
  std::vector<hipIpcMemHandle_t> handles;
  bool tag_reserved = false;
  local = pending ? Status::success() : bad(StatusCode::InvalidState, "alloc", "descriptor allocation failed");
  if (local.ok()) try {
    pending->tag = tag;
    pending->nbytes = nbytes;
    pending->memory_kind = kind;
    pending->id = active->next_allocation_id;
    pending->generation = active->generation;
    pending->host_peer_bases.resize(active->npes, nullptr);
    pending->imports_open.resize(active->npes, false);
    active->descriptors.reserve(active->descriptors.size() + 1);
    active->live_tags.reserve(active->live_tags.size() + 1);
    handles.resize(active->npes);
    active->live_tags.insert(pending->tag);
    tag_reserved = true;
  } catch (const std::bad_alloc&) {
    local = bad(StatusCode::InvalidState, "alloc", "host metadata allocation failed");
  }
  status = agree(active, local, "host descriptor preparation");
  if (!status.ok()) {
    if (tag_reserved) active->live_tags.erase(pending->tag);
    return status;
  }

  DeviceGuard guard;
  fatal_if_error(active, guard.enter(active->device_ordinal), "alloc device guard");
  if (injected("alloc")) local = injection("alloc", active);
  else if (kind == MemoryKind::DeviceDefault)
    local = hip_result(hipMalloc(&pending->owner_base, nbytes), "hipMalloc default");
  else
    local = hip_result(hipExtMallocWithFlags(&pending->owner_base, nbytes,
                                             hipDeviceMallocFinegrained), "hipExtMallocWithFlags fine-grained");
  if (local.ok()) ++pending->owner_allocated_count;
  status = agree(active, local, "local allocation");
  if (!status.ok()) { rollback(active, pending.get(), guard); active->live_tags.erase(pending->tag); return status; }
  pending->host_peer_bases[active->rank] = static_cast<std::byte*>(pending->owner_base);
  log_resource(active, "owner-allocated", pending.get());

  hipIpcMemHandle_t local_handle{};
  local = injected("export") ? injection("export", active) :
          hip_result(hipIpcGetMemHandle(&local_handle, pending->owner_base), "hipIpcGetMemHandle");
  status = agree(active, local, "export handle");
  if (!status.ok()) { rollback(active, pending.get(), guard); active->live_tags.erase(pending->tag); return status; }
  bootstrap::checked_mpi(MPI_Allgather(&local_handle, sizeof(local_handle), MPI_BYTE,
                                      handles.data(), sizeof(local_handle), MPI_BYTE, active->comm),
                         active->comm, "opaque handle exchange");
  log_resource(active, "handle-exported-and-exchanged", pending.get());

  local = Status::success();
  int opens = 0;
  for (int peer = 0; peer < active->npes; ++peer) if (peer != active->rank) {
    if (injected("open") && (active->npes == 2 || opens >= 1)) {
      local = injection("open", active); break;
    }
    void* imported = nullptr;
    local = hip_result(hipIpcOpenMemHandle(&imported, handles[peer],
                                           hipIpcMemLazyEnablePeerAccess), "hipIpcOpenMemHandle");
    if (!local.ok()) break;
    pending->host_peer_bases[peer] = static_cast<std::byte*>(imported);
    pending->imports_open[peer] = true;
    ++pending->opened_imports_count;
    ++opens;
  }
  status = agree(active, local, "open imports");
  if (!status.ok()) { rollback(active, pending.get(), guard); active->live_tags.erase(pending->tag); return status; }
  log_resource(active, "all-imports-open", pending.get());

  if (injected("table_alloc")) local = injection("table_alloc", active);
  else local = hip_result(hipMalloc(reinterpret_cast<void**>(&pending->device_peer_bases),
                                    std::size_t(active->npes) * sizeof(std::byte*)), "hipMalloc pointer table");
  status = agree(active, local, "pointer table allocation");
  if (!status.ok()) { rollback(active, pending.get(), guard); active->live_tags.erase(pending->tag); return status; }
  if (injected("table_upload")) local = injection("table_upload", active);
  else local = hip_result(hipMemcpy(pending->device_peer_bases, pending->host_peer_bases.data(),
                                    std::size_t(active->npes) * sizeof(std::byte*), hipMemcpyHostToDevice),
                          "hipMemcpy pointer table");
  if (local.ok()) local = hip_result(hipDeviceSynchronize(), "pointer table upload completion");
  status = agree(active, local, "pointer table upload");
  if (!status.ok()) { rollback(active, pending.get(), guard); active->live_tags.erase(pending->tag); return status; }

  // Commit only after all ranks have a complete immutable peer table.
  pending->live = true;
  Allocation* published = pending.get();
  active->descriptors.push_back(std::move(pending));
  ++active->live_allocations;
  ++active->next_allocation_id;
  log_resource(active, "committed", published);
  restore_or_abort(active, guard);
  *out = published;
  return Status::success();
}

void* local_ptr(const Allocation* allocation) {
  Context* ctx = active_context_ptr();
  if (!ctx || std::this_thread::get_id() != ctx->owner_thread) return nullptr;
  Allocation* a = find_descriptor(ctx, allocation);
  return a && a->live && a->generation == ctx->generation ? a->owner_base : nullptr;
}

Status remote_ptr(Context* ctx, const Allocation* allocation,
                  std::size_t byte_offset, std::size_t access_bytes,
                  int peer, void** out_device_ptr) {
  Context* active = active_context_ptr();
  if (!active || ctx != active) return bad(StatusCode::InvalidState, "remote_ptr", "invalid Context");
  if (std::this_thread::get_id() != active->owner_thread)
    return bad(StatusCode::InvalidState, "remote_ptr", "wrong thread");
  Allocation* a = find_descriptor(active, allocation);
  if (!a || !a->live || a->generation != active->generation)
    return bad(StatusCode::InvalidState, "remote_ptr", "unknown or freed allocation");
  if (!out_device_ptr) return bad(StatusCode::InvalidArgument, "remote_ptr", "null output");
  auto status = metadata::validate_peer(peer, active->npes);
  if (!status.ok()) return status;
  status = metadata::validate_range(a->nbytes, byte_offset, access_bytes);
  if (!status.ok()) return status;
  auto base = reinterpret_cast<std::uintptr_t>(a->host_peer_bases[peer]);
  if (!base || base > std::numeric_limits<std::uintptr_t>::max() - byte_offset)
    return bad(StatusCode::InvalidState, "remote_ptr", "mapped address unavailable or overflow");
  *out_device_ptr = reinterpret_cast<void*>(base + byte_offset);
  return Status::success();
}

DeviceView device_view(const Allocation* allocation) {
  Context* ctx = active_context_ptr();
  if (!ctx || std::this_thread::get_id() != ctx->owner_thread) return {};
  Allocation* a = find_descriptor(ctx, allocation);
  if (!a || !a->live || a->generation != ctx->generation) return {};
  DeviceView view{};
  view.local_base = static_cast<std::byte*>(a->owner_base);
  view.peer_bases_device = a->device_peer_bases;
  view.nbytes = a->nbytes;
  view.my_pe = ctx->rank;
  view.npes = ctx->npes;
  view.allocation_id = a->id;
  view.generation = a->generation;
  return view;
}

Status symm_free(Context* ctx, Allocation** allocation) {
  Context* active = active_context_ptr();
  if (!active) return bad(StatusCode::InvalidState, "free", "no active Context");
  if (std::this_thread::get_id() != active->owner_thread)
    return bad(StatusCode::InvalidState, "free", "wrong thread; no MPI collective entered");
  Allocation* a = allocation ? find_descriptor(active, *allocation) : nullptr;
  Status local = ctx != active ? bad(StatusCode::InvalidState, "free", "invalid Context") :
                 !a || !a->live || a->generation != active->generation
                     ? bad(StatusCode::InvalidState, "free", "unknown or freed allocation") : Status::success();
  metadata::OperationHeader header{};
  auto built = metadata::make_header(metadata::OpKind::Free, active->sequence,
                                     a ? a->id : 0, active->generation,
                                     a ? a->tag.c_str() : "<invalid>",
                                     a ? a->nbytes : 1,
                                     a ? a->memory_kind : MemoryKind::DeviceDefault, &header);
  if (local.ok() && !built.ok()) local = built;
  auto status = collective_gate_header(ctx, header, local);
  if (!status.ok()) return status;

  DeviceGuard guard;
  fatal_if_error(active, guard.enter(active->device_ordinal), "free device guard");
  fatal_if_error(active, hip_result(hipDeviceSynchronize(), "free HIP completion"), "free synchronize");
  fatal_if_error(active, agree(active, Status::success(), "all GPU users complete"), "free sync agreement");
  log_resource(active, "all-gpu-users-complete", a);
  close_imports(active, a); // Includes a collective confirmation before owner release.
  free_owned_resources(active, a);
  restore_or_abort(active, guard);
  a->live = false;
  active->live_tags.erase(a->tag);
  --active->live_allocations;
  *allocation = nullptr;
  log_resource(active, "tombstoned", a);
  return Status::success();
}

} // namespace dcu_ipc
