// Thin TileLang C ABI over the embedded, verified dcu_ipc Context API.
// It owns only DeviceView upload buffers; all MPI/IPC allocation semantics
// remain in context.cpp/allocation.cpp.
#include <dcu_ipc/api.hpp>
#include <tilelang/dcu_ipc_runtime.h>
#include "context_internal.hpp"

#include <hip/hip_runtime.h>

#include <string>
#include <thread>
#include <unordered_map>

struct tl_dcu_ipc_context { dcu_ipc::Context* value; };
struct tl_dcu_ipc_allocation { dcu_ipc::Allocation* value; };

namespace {
thread_local std::string last_error;
std::unordered_map<dcu_ipc::Allocation*, std::uint64_t*> descriptors;

int status_code(const dcu_ipc::Status& status) {
  if (status.ok()) { last_error.clear(); return 0; }
  last_error = status.phase + ": " + status.detail;
  return static_cast<int>(status.code);
}
int argument_error(const char* detail) {
  last_error = detail;
  return static_cast<int>(dcu_ipc::StatusCode::InvalidArgument);
}
}  // namespace

extern "C" int tl_dcu_ipc_init_world(int ordinal, tl_dcu_ipc_context** out) {
  if (!out || *out) return argument_error("init output is null or already populated");
  dcu_ipc::Context* context = nullptr;
  dcu_ipc::InitOptions options;
  if (ordinal >= 0) { options.selection = dcu_ipc::DeviceSelection::ExplicitOrdinal; options.ordinal = ordinal; }
  int code = status_code(dcu_ipc::init_mpi(MPI_COMM_WORLD, options, &context));
  if (code) return code;
  *out = new tl_dcu_ipc_context{context};
  return 0;
}
extern "C" int tl_dcu_ipc_rank(const tl_dcu_ipc_context* context) { return context ? dcu_ipc::my_pe(context->value) : -1; }
extern "C" int tl_dcu_ipc_world_size(const tl_dcu_ipc_context* context) { return context ? dcu_ipc::num_pes(context->value) : -1; }
extern "C" int tl_dcu_ipc_alloc(tl_dcu_ipc_context* context, const char* tag, size_t bytes,
                                  int fine_grained, tl_dcu_ipc_allocation** out) {
  if (!context || !out || *out) return argument_error("alloc context/output invalid");
  dcu_ipc::Allocation* allocation = nullptr;
  int code = status_code(dcu_ipc::symm_alloc(context->value, tag, bytes,
      fine_grained ? dcu_ipc::MemoryKind::FineGrained : dcu_ipc::MemoryKind::DeviceDefault,
      &allocation));
  if (code) return code;
  *out = new tl_dcu_ipc_allocation{allocation};
  return 0;
}
extern "C" void* tl_dcu_ipc_local_ptr(const tl_dcu_ipc_allocation* allocation) {
  return allocation ? dcu_ipc::local_ptr(allocation->value) : nullptr;
}
extern "C" int tl_dcu_ipc_device_view_words(tl_dcu_ipc_context* context,
                                               const tl_dcu_ipc_allocation* allocation,
                                               uint64_t** out) {
  if (!context || !allocation || !out) return argument_error("descriptor context/allocation/output invalid");
  if (context->value != dcu_ipc::active_context_ptr() ||
      context->value->owner_thread != std::this_thread::get_id())
    return argument_error("descriptor requires the live context owner thread");
  auto found = descriptors.find(allocation->value);
  if (found != descriptors.end()) { *out = found->second; return 0; }
  const dcu_ipc::DeviceView view = dcu_ipc::device_view(allocation->value);
  if (!view.local_base || !view.peer_bases_device || view.nbytes == 0)
    return argument_error("descriptor requested for non-live allocation");
  static_assert(sizeof(dcu_ipc::DeviceView) == 6 * sizeof(std::uint64_t), "M2 descriptor ABI");
  dcu_ipc::DeviceGuard guard;
  auto selected = guard.enter(context->value->device_ordinal);
  if (!selected.ok()) return status_code(selected);
  std::uint64_t* device = nullptr;
  hipError_t hip = hipMalloc(reinterpret_cast<void**>(&device), sizeof(view));
  if (hip == hipSuccess) hip = hipMemcpy(device, &view, sizeof(view), hipMemcpyHostToDevice);
  if (hip != hipSuccess) {
    if (device) hipFree(device);
    auto restored = guard.restore();
    if (!restored.ok()) return status_code(restored);
    last_error = std::string("descriptor HIP upload: ") + hipGetErrorString(hip);
    return static_cast<int>(dcu_ipc::StatusCode::HipError);
  }
  auto restored = guard.restore();
  if (!restored.ok()) { hipFree(device); return status_code(restored); }
  descriptors.emplace(allocation->value, device);
  *out = device;
  return 0;
}
extern "C" int tl_dcu_ipc_sync(tl_dcu_ipc_context* context) {
  return context ? status_code(dcu_ipc::sync_all(context->value)) : argument_error("sync context invalid");
}
extern "C" int tl_dcu_ipc_free(tl_dcu_ipc_context* context, tl_dcu_ipc_allocation** allocation) {
  if (!context || !allocation || !*allocation) return argument_error("free context/allocation invalid");
  auto* wrapper = *allocation;
  // Preserve the descriptor until the embedded core completes its collective
  // liveness/import-close protocol.  In particular, a wrong-thread or failed
  // free must leave a still-live allocation's cached descriptor usable.
  auto* core_allocation = wrapper->value;
  int code = status_code(dcu_ipc::symm_free(context->value, &wrapper->value));
  if (code) return code;
  auto found = descriptors.find(core_allocation);
  if (found != descriptors.end()) {
    dcu_ipc::DeviceGuard guard;
    auto selected = guard.enter(context->value->device_ordinal);
    if (!selected.ok()) return status_code(selected);
    const hipError_t hip = hipFree(found->second);
    auto restored = guard.restore();
    if (!restored.ok()) return status_code(restored);
    if (hip != hipSuccess) {
      // The collective core allocation is already retired.  It is unsafe to
      // leave a dangling allocation key in the descriptor cache or pretend a
      // retry could launch through this descriptor; report the leak loudly
      // while invalidating the public allocation handle.
      descriptors.erase(found);
      delete wrapper;
      *allocation = nullptr;
      last_error = std::string("descriptor HIP free: ") + hipGetErrorString(hip);
      return static_cast<int>(dcu_ipc::StatusCode::HipError);
    }
    descriptors.erase(found);
  }
  delete wrapper; *allocation = nullptr;
  return 0;
}
extern "C" int tl_dcu_ipc_finalize(tl_dcu_ipc_context** context) {
  if (!context || !*context) return argument_error("finalize context invalid");
  // Do not make a local descriptor-cache decision before entering the core's
  // collective finalize gate.  A descriptor may intentionally have been
  // uploaded on only a subset of PEs; returning early on those ranks would
  // strand the remaining ranks inside the collective.  The embedded Context
  // owns the authoritative, symmetric live-allocation check and returns Busy
  // uniformly while any allocation remains live.
  auto* wrapper = *context;
  int code = status_code(dcu_ipc::finalize(&wrapper->value));
  if (code) return code;
  delete wrapper; *context = nullptr;
  return 0;
}
extern "C" const char* tl_dcu_ipc_last_error(void) { return last_error.c_str(); }
