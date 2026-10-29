#include "context_internal.hpp"
#include "bootstrap_mpi.hpp"

#include <hip/hip_runtime.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <vector>
#include <unistd.h>

namespace dcu_ipc {
namespace {
Context* active_context = nullptr;
std::uint64_t generation_counter = 0;
constexpr int max_visible_devices = 64;
struct DeviceList {
  int count;
  char bdfs[max_visible_devices][64];
};
struct Compatibility {
  std::uint32_t version;
  std::uint32_t header_bytes;
  std::uint32_t handle_bytes;
  std::uint32_t hip_runtime;
};
Status error(StatusCode code, const char* phase, std::string detail) {
  return Status::error(code, phase, std::move(detail));
}
Status hip_status(hipError_t rc, const char* phase) {
  if (rc == hipSuccess) return Status::success();
  return error(StatusCode::HipError, phase,
               std::to_string(int(rc)) + " " + hipGetErrorString(rc));
}
[[noreturn]] void abort_unrecoverable(MPI_Comm comm, const char* phase, const Status& status) {
  int rank = -1;
  if (comm != MPI_COMM_NULL) MPI_Comm_rank(comm, &rank);
  std::fprintf(stderr, "FATAL rank=%d phase=%s code=%d detail=%s\n", rank, phase,
               int(status.code), status.detail.c_str());
  std::fflush(stderr);
  MPI_Abort(comm == MPI_COMM_NULL ? MPI_COMM_WORLD : comm, 99);
  std::abort();
}
Status validate_mpi_entry(MPI_Comm comm) {
  int initialized = 0, finalized = 0, main_thread = 0, provided = 0;
  if (MPI_Initialized(&initialized) != MPI_SUCCESS || !initialized)
    return error(StatusCode::MpiError, "init", "MPI must already be initialized");
  if (MPI_Finalized(&finalized) != MPI_SUCCESS || finalized)
    return error(StatusCode::MpiError, "init", "MPI has been finalized");
  if (MPI_Is_thread_main(&main_thread) != MPI_SUCCESS || !main_thread)
    return error(StatusCode::MpiError, "init", "call must be on MPI main thread");
  if (MPI_Query_thread(&provided) != MPI_SUCCESS || provided < MPI_THREAD_FUNNELED)
    return error(StatusCode::MpiError, "init", "MPI_THREAD_FUNNELED required");
  if (comm == MPI_COMM_NULL)
    return error(StatusCode::InvalidArgument, "init", "MPI_COMM_NULL has no collective error path");
  int inter = 0;
  auto status = bootstrap::checked_mpi(MPI_Comm_test_inter(comm, &inter), comm, "comm type");
  if (!status.ok()) return status;
  if (inter) return error(StatusCode::InvalidArgument, "init", "intercommunicator unsupported");
  return Status::success();
}
Status validate_active(Context* supplied) {
  if (!active_context) return error(StatusCode::InvalidState, "context", "no active Context");
  if (supplied != active_context)
    return error(StatusCode::InvalidState, "context", "null or foreign Context handle");
  if (std::this_thread::get_id() != active_context->owner_thread)
    return error(StatusCode::InvalidState, "context", "wrong host thread");
  return Status::success();
}
Status clean_comm(MPI_Comm* comm) {
  if (*comm != MPI_COMM_NULL)
    return bootstrap::checked_mpi(MPI_Comm_free(comm), MPI_COMM_WORLD, "cleanup communicator");
  return Status::success();
}
}

Status DeviceGuard::enter(int selected) {
  if (previous_ >= 0) return error(StatusCode::InvalidState, "device guard", "already entered");
  auto status = hip_status(hipGetDevice(&previous_), "hipGetDevice");
  if (!status.ok()) return status;
  changed_ = previous_ != selected;
  if (changed_) {
    status = hip_status(hipSetDevice(selected), "hipSetDevice");
    if (!status.ok()) { previous_ = -1; changed_ = false; }
  }
  return status;
}
Status DeviceGuard::restore() {
  if (previous_ < 0) return error(StatusCode::InvalidState, "device guard", "not entered");
  auto status = changed_ ? hip_status(hipSetDevice(previous_), "restore device") : Status::success();
  previous_ = -1;
  changed_ = false;
  return status;
}

Context* active_context_ptr() noexcept { return active_context; }
[[noreturn]] void abort_context(MPI_Comm comm, const char* phase, const Status& status) {
  abort_unrecoverable(comm, phase, status);
}

Status collective_gate_header(Context* supplied, const metadata::OperationHeader& header,
                              const Status& local) {
  if (!active_context) return error(StatusCode::InvalidState, "gate", "no active Context");
  if (std::this_thread::get_id() != active_context->owner_thread)
    return error(StatusCode::InvalidState, "gate", "wrong host thread; no MPI collective entered");
  auto entry = validate_active(supplied);
  return bootstrap::operation_gate(active_context->comm, &active_context->sequence,
                                   header, entry.ok() ? local : entry);
}

Status collective_gate(Context* supplied, metadata::OpKind kind, const Status& local) {
  if (!active_context) return error(StatusCode::InvalidState, "gate", "no active Context");
  if (std::this_thread::get_id() != active_context->owner_thread)
    return error(StatusCode::InvalidState, "gate", "wrong host thread; no MPI collective entered");
  auto entry = validate_active(supplied);
  metadata::OperationHeader header{};
  auto built = metadata::make_header(kind, active_context->sequence, 0,
                                     active_context->generation, nullptr, 0,
                                     MemoryKind::DeviceDefault, &header);
  return collective_gate_header(supplied, header, !entry.ok() ? entry : !built.ok() ? built : local);
}

Status init_mpi(MPI_Comm comm, const InitOptions& options, Context** out) {
  auto entry = validate_mpi_entry(comm);
  if (!entry.ok()) return entry; // Invalid communicator/thread has no safe collective path.
  Status local = metadata::validate_init_options(options);
  if (local.ok() && !out) local = error(StatusCode::InvalidArgument, "init", "null out pointer");
  if (local.ok() && *out) local = error(StatusCode::InvalidState, "init", "out pointer already populated");
  if (local.ok() && active_context) local = error(StatusCode::InvalidState, "init", "Context already active");
  auto status = bootstrap::collective_status(comm, local, "init arguments");
  if (!status.ok()) return status;

  MPI_Comm owned = MPI_COMM_NULL, shared = MPI_COMM_NULL;
  status = bootstrap::checked_mpi(MPI_Comm_dup(comm, &owned), comm, "duplicate communicator");
  if (!status.ok()) return status;
  status = bootstrap::checked_mpi(MPI_Comm_set_errhandler(owned, MPI_ERRORS_RETURN), owned,
                                  "set communicator errors-return");
  if (!status.ok()) return status;
  int rank = -1, npes = 0, local_rank = -1, local_size = 0;
  bootstrap::checked_mpi(MPI_Comm_rank(owned, &rank), owned, "rank");
  bootstrap::checked_mpi(MPI_Comm_size(owned, &npes), owned, "size");
  bootstrap::checked_mpi(MPI_Comm_split_type(owned, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shared),
                         owned, "shared split");
  bootstrap::checked_mpi(MPI_Comm_rank(shared, &local_rank), owned, "local rank");
  bootstrap::checked_mpi(MPI_Comm_size(shared, &local_size), owned, "local size");
  status = bootstrap::collective_status(owned,
      local_size == npes ? Status::success() : error(StatusCode::MultiNodeUnsupported, "init", "communicator crosses nodes"),
      "single-node check");
  clean_comm(&shared);
  if (!status.ok()) { clean_comm(&owned); return status; }

  Compatibility mine{};
  mine.version = metadata::protocol_version;
  mine.header_bytes = sizeof(metadata::OperationHeader);
  mine.handle_bytes = sizeof(hipIpcMemHandle_t);
  int runtime = 0;
  local = hip_status(hipRuntimeGetVersion(&runtime), "hipRuntimeGetVersion");
  status = bootstrap::collective_status(owned, local, "runtime version query");
  if (!status.ok()) { clean_comm(&owned); return status; }
  mine.hip_runtime = std::uint32_t(runtime);
  std::vector<Compatibility> versions(npes);
  bootstrap::checked_mpi(MPI_Allgather(&mine, sizeof(mine), MPI_BYTE, versions.data(),
                                      sizeof(mine), MPI_BYTE, owned), owned, "compatibility gather");
  local = Status::success();
  for (int p = 0; p < npes; ++p) {
    if (std::memcmp(&versions[p], &mine, sizeof(mine)) != 0) {
      local = error(StatusCode::SymmetryMismatch, "init", "protocol/runtime/handle size mismatch at rank " + std::to_string(p));
      break;
    }
  }
  status = bootstrap::collective_status(owned, local, "compatibility check");
  if (!status.ok()) { clean_comm(&owned); return status; }

  DeviceList visible{};
  local = hip_status(hipGetDeviceCount(&visible.count), "hipGetDeviceCount");
  if (local.ok() && (visible.count < 1 || visible.count > max_visible_devices))
    local = error(StatusCode::DeviceSelection, "init", "visible device count outside 1..64");
  if (local.ok()) for (int i = 0; i < visible.count; ++i) {
    local = hip_status(hipDeviceGetPCIBusId(visible.bdfs[i], sizeof(visible.bdfs[i]), i),
                       "hipDeviceGetPCIBusId");
    if (!local.ok()) break;
    if (visible.bdfs[i][0] == '\0') {
      local = error(StatusCode::DeviceSelection, "init", "empty BDF identity");
      break;
    }
  }
  status = bootstrap::collective_status(owned, local, "device inventory");
  if (!status.ok()) { clean_comm(&owned); return status; }
  std::vector<DeviceList> lists(npes);
  bootstrap::checked_mpi(MPI_Allgather(&visible, sizeof(visible), MPI_BYTE,
                                      lists.data(), sizeof(visible), MPI_BYTE, owned),
                         owned, "visible device lists");
  bool same_list = true;
  for (int p = 1; p < npes; ++p)
    if (lists[p].count != lists[0].count ||
        std::memcmp(lists[p].bdfs, lists[0].bdfs, sizeof(visible.bdfs)) != 0)
      same_list = false;
  int selected = -1;
  if (options.selection == DeviceSelection::ExplicitOrdinal) selected = options.ordinal;
  else if (visible.count == 1) selected = 0;
  else if (same_list && visible.count >= npes) selected = local_rank;
  else local = error(StatusCode::DeviceSelection, "init", "ambiguous auto mapping; supply explicit ordinal");
  if (local.ok() && (selected < 0 || selected >= visible.count))
    local = error(StatusCode::DeviceSelection, "init", "chosen ordinal outside visible devices");
  status = bootstrap::collective_status(owned, local, "device selection");
  if (!status.ok()) { clean_comm(&owned); return status; }

  std::array<char, 64> selected_bdf{};
  std::strncpy(selected_bdf.data(), visible.bdfs[selected], selected_bdf.size() - 1);
  std::vector<std::array<char, 64>> all_bdfs(npes);
  bootstrap::checked_mpi(MPI_Allgather(selected_bdf.data(), selected_bdf.size(), MPI_BYTE,
                                      all_bdfs.data(), selected_bdf.size(), MPI_BYTE, owned),
                         owned, "selected BDF gather");
  local = Status::success();
  for (int a = 0; a < npes; ++a) for (int b = a + 1; b < npes; ++b)
    if (std::strcmp(all_bdfs[a].data(), all_bdfs[b].data()) == 0)
      local = error(StatusCode::DeviceSelection, "init", "duplicate physical BDF at ranks " +
                    std::to_string(a) + "," + std::to_string(b));
  status = bootstrap::collective_status(owned, local, "physical identity uniqueness");
  if (!status.ok()) { clean_comm(&owned); return status; }

  DeviceGuard guard;
  local = guard.enter(selected);
  const bool guard_entered = local.ok();
  hipDeviceProp_t properties{};
  if (local.ok()) local = hip_status(hipGetDeviceProperties(&properties, selected), "hipGetDeviceProperties");
  status = bootstrap::collective_status(owned, local, "selected device properties");
  if (!status.ok()) {
    if (guard_entered) {
      auto restored = guard.restore();
      if (!restored.ok()) abort_unrecoverable(owned, "restore after failed init", restored);
    }
    clean_comm(&owned); return status;
  }

  std::uint64_t generation = 0;
  if (rank == 0) {
    if (generation_counter == std::numeric_limits<std::uint64_t>::max())
      abort_unrecoverable(owned, "generation", error(StatusCode::InvalidState, "init", "generation exhausted"));
    generation = ++generation_counter;
  }
  bootstrap::checked_mpi(MPI_Bcast(&generation, 1, MPI_UINT64_T, 0, owned), owned, "generation broadcast");
  auto* context = new (std::nothrow) Context;
  local = context ? Status::success() : error(StatusCode::InvalidState, "init", "Context allocation failed");
  status = bootstrap::collective_status(owned, local, "Context allocation");
  if (!status.ok()) {
    delete context;
    auto restored = guard.restore();
    if (!restored.ok()) abort_unrecoverable(owned, "restore after failed Context allocation", restored);
    clean_comm(&owned); return status;
  }
  context->comm = owned;
  context->rank = rank;
  context->npes = npes;
  context->local_rank = local_rank;
  context->device_ordinal = selected;
  context->pci_bdf = selected_bdf.data();
  context->arch = properties.gcnArchName;
  context->generation = generation;
  context->owner_thread = std::this_thread::get_id();
  guard.keep_selected(); // Successful init binds the caller to its selected device.
  active_context = context;
  *out = context;
  char host[256]{};
  if (gethostname(host, sizeof(host) - 1) != 0) std::strncpy(host, "unknown", sizeof(host) - 1);
  std::printf("CONTEXT rank=%d npes=%d local_rank=%d host=%s ordinal=%d bdf=%s arch=%s generation=%llu runtime=%u handle_bytes=%u\n",
              rank, npes, local_rank, host, selected, context->pci_bdf.c_str(),
              context->arch.c_str(), static_cast<unsigned long long>(generation),
              mine.hip_runtime, mine.handle_bytes);
  std::fflush(stdout);
  return Status::success();
}

int my_pe(const Context* ctx) { return ctx && ctx == active_context ? ctx->rank : -1; }
int num_pes(const Context* ctx) { return ctx && ctx == active_context ? ctx->npes : -1; }

Status sync_all(Context* ctx) {
  if (!active_context) return error(StatusCode::InvalidState, "sync", "no active Context");
  auto status = collective_gate(ctx, metadata::OpKind::Sync, Status::success());
  if (!status.ok()) return status;
  DeviceGuard guard;
  Status local = guard.enter(active_context->device_ordinal);
  if (local.ok()) {
    local = hip_status(hipDeviceSynchronize(), "hipDeviceSynchronize");
    Status restored = guard.restore();
    if (local.ok()) local = restored;
  }
  if (!local.ok()) abort_unrecoverable(active_context->comm, "sync", local);
  status = bootstrap::collective_status(active_context->comm, local, "sync HIP complete");
  if (!status.ok()) return status;
  return bootstrap::checked_mpi(MPI_Barrier(active_context->comm), active_context->comm, "sync barrier");
}

Status finalize(Context** ctx) {
  if (!active_context) return error(StatusCode::InvalidState, "finalize", "no active Context");
  Status local = !ctx || *ctx != active_context
      ? error(StatusCode::InvalidState, "finalize", "null or foreign Context handle")
      : Status::success();
  if (local.ok() && active_context->live_allocations)
    local = error(StatusCode::Busy, "finalize", "live allocations remain");
  auto status = collective_gate(ctx ? *ctx : nullptr, metadata::OpKind::Finalize, local);
  if (!status.ok()) return status;
  Context* retiring = active_context;
  status = bootstrap::checked_mpi(MPI_Comm_free(&retiring->comm), MPI_COMM_WORLD, "free Context communicator");
  if (!status.ok()) return status;
  active_context = nullptr;
  *ctx = nullptr;
  delete retiring;
  return Status::success();
}

} // namespace dcu_ipc
