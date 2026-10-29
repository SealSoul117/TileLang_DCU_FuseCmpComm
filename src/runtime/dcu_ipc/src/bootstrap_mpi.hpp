#pragma once

#include <dcu_ipc/types.hpp>
#include "metadata.hpp"
#include <mpi.h>

namespace dcu_ipc::bootstrap {

// MPI failures after a valid communicator is established are unrecoverable:
// abort the communicator so peer ranks do not wait forever.
Status checked_mpi(int rc, MPI_Comm comm, const char* phase);
Status collective_status(MPI_Comm comm, const Status& local, const char* phase);
Status operation_gate(MPI_Comm comm, std::uint64_t* sequence,
                      const metadata::OperationHeader& header, const Status& local);

} // namespace dcu_ipc::bootstrap
