/*!
 * \file tl/op/distributed.cc
 * \brief M1 distributed TIR operation registration.
 *
 * These operations intentionally have no HIP lowering in M1.  Their explicit
 * registry identity and effect kinds keep communication/synchronization from
 * being folded or moved as an untyped call_extern string.  M2 supplies the
 * embedded HIP-IPC device implementation.
 */

#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

namespace tvm {
namespace tl {
using namespace tir;

#define TL_REGISTER_DISTRIBUTED_OP(Name, Inputs, Effect, Printer)             \
  TVM_REGISTER_OP(Name)                                                        \
      .set_num_inputs(Inputs)                                                  \
      .set_attr<TCallEffectKind>("TCallEffectKind", Integer(Effect))         \
      .set_attr<TScriptPrinterName>("TScriptPrinterName", Printer)

// Rank/world-size are runtime-context reads, never compile-process constants.
TL_REGISTER_DISTRIBUTED_OP("tl.get_rank", 0, CallEffectKind::kOpaque, "get_rank");
TL_REGISTER_DISTRIBUTED_OP("tl.get_num_ranks", 0, CallEffectKind::kOpaque,
                           "get_num_ranks");

// A serializable frontend-only marker.  NormalizeRemoteViews consumes it.
TL_REGISTER_DISTRIBUTED_OP("tl.remote_view_token", -1, CallEffectKind::kEmbedInfo,
                           "remote_view_token");
// M1 normal form: `Let(remote_ptr, tl.remote_address(provenance, peer), ...)`.
// The address denotes the allocation base; Buffer.elem_offset supplies the
// logical view offset exactly once.
TL_REGISTER_DISTRIBUTED_OP("tl.remote_address", 2, CallEffectKind::kOpaque,
                           "remote_address");
TL_REGISTER_DISTRIBUTED_OP("tl.ipc.remote_address", 4, CallEffectKind::kOpaque,
                           "ipc_remote_address");
// M2 pre-SplitHostDevice form.  These calls carry the DeviceView descriptor
// and rank/world ABI values as ordinary operands, rather than relying on a
// PrimFunc attribute that host/device splitting could silently discard.
TL_REGISTER_DISTRIBUTED_OP("tl.ipc.fence_sys", 2, CallEffectKind::kOpaque,
                           "ipc_fence_sys");
// IPC publication uses a CTA rendezvous before one thread releases a flag.
// This explicit form is introduced only for bind_ipc functions so HIP can
// preserve the required whole-block ordering without changing ordinary
// TileLang/GEMM storage-sync policy.
TL_REGISTER_DISTRIBUTED_OP("tl.ipc.block_barrier", 2, CallEffectKind::kOpaque,
                           "ipc_block_barrier");
TL_REGISTER_DISTRIBUTED_OP("tl.ipc.signal_set", 6, CallEffectKind::kOpaque,
                           "ipc_signal_set");
TL_REGISTER_DISTRIBUTED_OP("tl.ipc.wait", 9, CallEffectKind::kOpaque,
                           "ipc_wait");
TL_REGISTER_DISTRIBUTED_OP("tl.ipc.put_block", 7, CallEffectKind::kOpaque,
                           "ipc_put_block");
TL_REGISTER_DISTRIBUTED_OP("tl.ipc.get_block", 7, CallEffectKind::kOpaque,
                           "ipc_get_block");
TL_REGISTER_DISTRIBUTED_OP("tl.ipc.st", 8, CallEffectKind::kOpaque, "ipc_st");
TL_REGISTER_DISTRIBUTED_OP("tl.ipc.ld", 8, CallEffectKind::kOpaque, "ipc_ld");

TL_REGISTER_DISTRIBUTED_OP("tl.fence_sys", 0, CallEffectKind::kOpaque, "fence_sys");
TL_REGISTER_DISTRIBUTED_OP("tl.signal_set", 3, CallEffectKind::kOpaque, "signal_set");
TL_REGISTER_DISTRIBUTED_OP("tl.tileop.wait", 6, CallEffectKind::kOpaque, "wait");
TL_REGISTER_DISTRIBUTED_OP("tl.tileop.put_block", 4, CallEffectKind::kOpaque,
                           "put_block");
TL_REGISTER_DISTRIBUTED_OP("tl.tileop.get_block", 4, CallEffectKind::kOpaque,
                           "get_block");
TL_REGISTER_DISTRIBUTED_OP("tl.tileop.st", 5, CallEffectKind::kOpaque, "st");
TL_REGISTER_DISTRIBUTED_OP("tl.tileop.ld", 5, CallEffectKind::kOpaque, "ld");

#undef TL_REGISTER_DISTRIBUTED_OP

}  // namespace tl
}  // namespace tvm
