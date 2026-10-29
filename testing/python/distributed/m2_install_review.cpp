// Independent installed-component smoke test.  Compile only with the install
// prefix's headers/library plus system HIP/MPI, not with the source checkout.
#include <dcu_ipc/api.hpp>
#include <tilelang/dcu_ipc_runtime.h>
#include <hip/hip_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace {
int rank_id = -1;
void require(bool condition, const char* message) {
  if (condition) return;
  std::fprintf(stderr, "FAIL rank=%d: %s\n", rank_id, message);
  MPI_Abort(MPI_COMM_WORLD, 17);
  std::abort();
}
}

int main(int argc, char** argv) {
  int provided = 0;
  require(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) == MPI_SUCCESS,
          "MPI_Init_thread");
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
  require(provided >= MPI_THREAD_FUNNELED, "MPI thread support");
  dcu_ipc::Context* context = nullptr;
  require(dcu_ipc::init_mpi(MPI_COMM_WORLD, {}, &context).ok(), "init_mpi");
  require(dcu_ipc::my_pe(context) == rank_id, "rank binding");
  const int world = dcu_ipc::num_pes(context);
  require(world == 4, "four ranks required");

  dcu_ipc::Allocation* allocation = nullptr;
  auto mismatch = dcu_ipc::symm_alloc(context, "size-mismatch",
      rank_id == 0 ? 32 : 64, dcu_ipc::MemoryKind::FineGrained, &allocation);
  require(!mismatch.ok() && allocation == nullptr, "collective size mismatch rejected");
  require(dcu_ipc::symm_alloc(context, "installed-allocation", 4096,
      dcu_ipc::MemoryKind::FineGrained, &allocation).ok(), "allocation after rejected attempt");
  const auto view = dcu_ipc::device_view(allocation);
  require(view.my_pe == rank_id && view.npes == world && view.nbytes == 4096,
          "live allocation descriptor");
  require(view.local_base && view.peer_bases_device && view.allocation_id && view.generation,
          "descriptor fields");
  void* pointer = nullptr;
  require(dcu_ipc::remote_ptr(context, allocation, 12, 4, (rank_id + 1) % world,
                              &pointer).ok() && pointer, "remote address with byte offset");
  require(!dcu_ipc::remote_ptr(context, allocation, 4095, 4, 0, &pointer).ok(),
          "range overrun rejected");
  require(!dcu_ipc::remote_ptr(context, allocation, 0, 4, world, &pointer).ok(),
          "invalid peer rejected");
  require(!dcu_ipc::finalize(&context).ok() && context, "live allocation prevents finalize");
  require(dcu_ipc::sync_all(context).ok(), "collective completion");
  auto* stale = allocation;
  require(dcu_ipc::symm_free(context, &allocation).ok() && !allocation, "collective free");
  require(!dcu_ipc::local_ptr(stale) && !dcu_ipc::device_view(stale).local_base,
          "retired allocation rejected");
  require(dcu_ipc::finalize(&context).ok() && !context, "finalize");

  // Exercise the installed C ABI separately; TileLang must not own MPI.
  tl_dcu_ipc_context* ccontext = nullptr;
  require(tl_dcu_ipc_init_world(-1, &ccontext) == 0, "C ABI init");
  require(tl_dcu_ipc_rank(ccontext) == rank_id &&
          tl_dcu_ipc_world_size(ccontext) == world, "C ABI rank/world");
  tl_dcu_ipc_allocation* partial_descriptor = nullptr;
  require(tl_dcu_ipc_alloc(ccontext, "partial-descriptor", 4096, 1,
                           &partial_descriptor) == 0, "partial-descriptor allocation");
  if (rank_id == 0) {
    std::uint64_t* only_rank_zero = nullptr;
    require(tl_dcu_ipc_device_view_words(ccontext, partial_descriptor,
                                        &only_rank_zero) == 0,
            "descriptor need not be materialized on every rank");
  }
  require(tl_dcu_ipc_finalize(&ccontext) != 0 && ccontext,
          "collective Busy with asymmetric local descriptor caches");
  require(tl_dcu_ipc_free(ccontext, &partial_descriptor) == 0 &&
          !partial_descriptor, "free after rejected collective finalize");
  tl_dcu_ipc_allocation* calloc = nullptr;
  require(tl_dcu_ipc_alloc(ccontext, "cabi", 4096, 1, &calloc) == 0, "C ABI alloc");
  std::uint64_t* device_words = nullptr;
  int selected_device = -1, device_count = 0;
  require(hipGetDevice(&selected_device) == hipSuccess &&
          hipGetDeviceCount(&device_count) == hipSuccess, "current device query");
  const int other_device = (selected_device + 1) % device_count;
  require(hipSetDevice(other_device) == hipSuccess, "caller changes current device");
  require(tl_dcu_ipc_device_view_words(ccontext, calloc, &device_words) == 0 && device_words,
          "C ABI device descriptor");
  hipPointerAttribute_t attributes{};
  require(hipPointerGetAttributes(&attributes, device_words) == hipSuccess &&
          attributes.device == selected_device, "descriptor resides on context device");
  int restored_device = -1;
  require(hipGetDevice(&restored_device) == hipSuccess && restored_device == other_device,
          "descriptor creation preserves caller current device");
  require(hipSetDevice(selected_device) == hipSuccess, "caller restores selected device");
  int wrong_thread_status = 0;
  std::thread wrong_thread([&] {
    std::uint64_t* rejected = nullptr;
    wrong_thread_status = tl_dcu_ipc_device_view_words(ccontext, calloc, &rejected);
  });
  wrong_thread.join();
  require(wrong_thread_status != 0, "cached descriptor rejects wrong host thread");
  std::thread wrong_free_thread([&] {
    wrong_thread_status = tl_dcu_ipc_free(ccontext, &calloc);
  });
  wrong_free_thread.join();
  require(wrong_thread_status != 0 && calloc,
          "wrong-thread free must fail without retiring allocation");
  std::uint64_t* unchanged_words = nullptr;
  require(tl_dcu_ipc_device_view_words(ccontext, calloc, &unchanged_words) == 0 &&
          unchanged_words == device_words,
          "failed free preserves the device descriptor");
  dcu_ipc::DeviceView downloaded{};
  require(hipMemcpy(&downloaded, device_words, sizeof(downloaded), hipMemcpyDeviceToHost)
          == hipSuccess, "descriptor download");
  require(downloaded.my_pe == rank_id && downloaded.npes == world &&
          downloaded.local_base == tl_dcu_ipc_local_ptr(calloc), "C ABI descriptor consistency");
  require(tl_dcu_ipc_sync(ccontext) == 0, "C ABI sync");
  require(tl_dcu_ipc_free(ccontext, &calloc) == 0 && !calloc, "C ABI free");
  require(tl_dcu_ipc_finalize(&ccontext) == 0 && !ccontext, "C ABI finalize");
  int finalized = -1;
  require(MPI_Finalized(&finalized) == MPI_SUCCESS && finalized == 0,
          "runtime must not finalize caller MPI");
  std::printf("PASS installed-component rank=%d world=%d\n", rank_id, world);
  return MPI_Finalize();
}
