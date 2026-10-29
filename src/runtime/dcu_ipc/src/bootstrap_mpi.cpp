#include "bootstrap_mpi.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace dcu_ipc::bootstrap {
namespace {
struct WireStatus {
  std::uint32_t code;
  char phase[64];
  char detail[192];
};
void copy_text(char* out, std::size_t capacity, const std::string& text) {
  std::snprintf(out, capacity, "%s", text.c_str());
}
}

Status checked_mpi(int rc, MPI_Comm comm, const char* phase) {
  if (rc == MPI_SUCCESS) return Status::success();
  int rank = -1;
  if (comm != MPI_COMM_NULL) MPI_Comm_rank(comm, &rank);
  std::fprintf(stderr, "MPI FATAL rank=%d phase=%s code=%d\n", rank, phase, rc);
  std::fflush(stderr);
  MPI_Abort(comm == MPI_COMM_NULL ? MPI_COMM_WORLD : comm, rc);
  std::abort();
}

Status collective_status(MPI_Comm comm, const Status& local, const char* phase) {
  int size = 0;
  auto status = checked_mpi(MPI_Comm_size(comm, &size), comm, "status size");
  if (!status.ok()) return status;
  WireStatus mine{};
  mine.code = static_cast<std::uint32_t>(local.code);
  copy_text(mine.phase, sizeof(mine.phase), local.phase);
  copy_text(mine.detail, sizeof(mine.detail), local.detail);
  std::vector<WireStatus> all(size);
  status = checked_mpi(MPI_Allgather(&mine, sizeof(mine), MPI_BYTE, all.data(),
                                     sizeof(mine), MPI_BYTE, comm), comm, phase);
  if (!status.ok()) return status;
  for (int rank = 0; rank < size; ++rank) {
    if (all[rank].code > std::uint32_t(StatusCode::Busy) ||
        !std::memchr(all[rank].phase, '\0', sizeof(all[rank].phase)) ||
        !std::memchr(all[rank].detail, '\0', sizeof(all[rank].detail)))
      return Status::error(StatusCode::SymmetryMismatch, phase,
                           "malformed status at rank " + std::to_string(rank));
    if (all[rank].code != std::uint32_t(StatusCode::Ok))
      return Status::error(static_cast<StatusCode>(all[rank].code),
                           all[rank].phase[0] ? all[rank].phase : phase,
                           "rank " + std::to_string(rank) + ": " + all[rank].detail);
  }
  return Status::success();
}

Status operation_gate(MPI_Comm comm, std::uint64_t* sequence,
                      const metadata::OperationHeader& mine, const Status& local) {
  if (!sequence) return Status::error(StatusCode::InvalidState, "gate", "null sequence");
  int size = 0;
  auto status = checked_mpi(MPI_Comm_size(comm, &size), comm, "gate size");
  if (!status.ok()) return status;
  std::vector<metadata::OperationHeader> all(size);
  status = checked_mpi(MPI_Allgather(&mine, sizeof(mine), MPI_BYTE, all.data(),
                                     sizeof(mine), MPI_BYTE, comm), comm, "operation gate");
  if (!status.ok()) return status;
  ++*sequence; // Consume the attempt even when kinds mismatch; peers can retry together.
  auto header_status = metadata::validate_collective_headers(all);
  auto local_status = collective_status(comm,
      !local.ok() || mine.sequence + 1 == *sequence ? local :
      Status::error(StatusCode::InvalidState, "gate", "local operation sequence invalid"),
      "operation local status");
  if (!local_status.ok()) return local_status;
  return header_status;
}

} // namespace dcu_ipc::bootstrap
