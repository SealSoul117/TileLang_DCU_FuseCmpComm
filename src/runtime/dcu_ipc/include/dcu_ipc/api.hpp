#pragma once

#include <dcu_ipc/types.hpp>
#include <mpi.h>

namespace dcu_ipc {

struct Context;
struct Allocation;

// MPI_Init_thread(MPI_THREAD_FUNNELED) is owned by the application. init_mpi,
// symm_alloc, sync_all, symm_free and finalize are collective in identical
// order; my_pe, num_pes, local_ptr, remote_ptr and device_view are local queries.
// Successful init leaves the selected HIP ordinal current. Later collective
// calls temporarily bind it and restore a caller's different current ordinal.
Status init_mpi(MPI_Comm comm, const InitOptions& options, Context** out);
int my_pe(const Context* ctx);
int num_pes(const Context* ctx);
Status symm_alloc(Context* ctx, const char* tag, std::size_t nbytes,
                  MemoryKind kind, Allocation** out);
void* local_ptr(const Allocation* allocation);
Status remote_ptr(Context* ctx, const Allocation* allocation,
                  std::size_t byte_offset, std::size_t access_bytes,
                  int peer, void** out_device_ptr);
DeviceView device_view(const Allocation* allocation);
Status sync_all(Context* ctx); // Local HIP completion, then collective stage boundary.
Status symm_free(Context* ctx, Allocation** allocation); // Imports close before owner free.
Status finalize(Context** ctx); // Fails Busy while allocations remain live.

} // namespace dcu_ipc
