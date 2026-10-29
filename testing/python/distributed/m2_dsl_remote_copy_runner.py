"""Run a real public-DSL remote_view kernel on four embedded IPC peers.

This is intentionally a Python/MPI runner rather than a handwritten HIP
communication kernel: the executed device code is produced by
``tilelang.compile`` from the PrimFunc below and launched through its ctypes
adapter with runtime-provided DeviceView arguments.
"""
import ctypes

import tilelang
from tilelang import language as T
from tilelang.distributed.ipc_binding import bind_ipc

tilelang.disable_cache()


# The current public T.copy layout path admits this CTA-sized tile.  Payload
# values remain deliberately non-uniform; the block-publication runner adds
# unequal per-edge lengths through explicit put/get regions.
COUNT = 256


@T.prim_func
def dsl_remote_copy(src: T.Tensor((COUNT,), "uint32"), dst: T.Tensor((COUNT,), "uint32")):
    with T.Kernel(1, threads=256):
        remote = T.remote_view(dst, (T.get_rank() + 1) % T.get_num_ranks())
        T.copy(src, remote)


@T.prim_func
def dsl_get_block(src: T.Tensor((COUNT,), "uint32"), dst: T.Tensor((COUNT,), "uint32")):
    """Public cooperative get: pull the next rank's source into local dst."""
    with T.Kernel(1, threads=256):
        peer = (T.get_rank() + 1) % T.get_num_ranks()
        remote_src = T.remote_view(src, peer)
        T.get_block(remote_src[0], dst[0], COUNT, peer=peer)


def hip():
    library = ctypes.CDLL("libamdhip64.so")
    library.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    library.hipMemcpy.restype = ctypes.c_int
    library.hipDeviceSynchronize.restype = ctypes.c_int
    return library


def memcpy(api, dst, src, nbytes, kind):
    result = api.hipMemcpy(ctypes.c_void_p(dst), ctypes.c_void_p(src), nbytes, kind)
    if result:
        raise RuntimeError(f"hipMemcpy failed: {result}")


mpi = ctypes.CDLL("libmpi.so")
provided = ctypes.c_int()
if mpi.MPI_Init_thread(None, None, 1, ctypes.byref(provided)):
    raise RuntimeError("MPI_Init_thread failed")

from tilelang.distributed.ipc_runtime import IPCContext

context = IPCContext()
assert context.world_size == 4
src = context.alloc("dsl-copy-src", COUNT * 4)
dst = context.alloc("dsl-copy-dst", COUNT * 4)
api = hip()
host_source = (ctypes.c_uint32 * COUNT)(*[(context.rank + 1) * 100000 + i * 17 + 3 for i in range(COUNT)])
host_zero = (ctypes.c_uint32 * COUNT)()
memcpy(api, src.data_ptr(), ctypes.addressof(host_source), COUNT * 4, 1)  # HostToDevice
memcpy(api, dst.data_ptr(), ctypes.addressof(host_zero), COUNT * 4, 1)
if api.hipDeviceSynchronize():
    raise RuntimeError("source initialization did not synchronize")
context.sync()

kernel = tilelang.compile(bind_ipc(dsl_remote_copy, ["dst"]), target="hip -mcpu=gfx936", execution_backend="ctypes")
kernel(src, dst, dst.device_view(), context.rank, context.world_size)
if api.hipDeviceSynchronize():
    raise RuntimeError("DSL remote copy did not synchronize")
context.sync()
result = (ctypes.c_uint32 * COUNT)()
memcpy(api, ctypes.addressof(result), dst.data_ptr(), COUNT * 4, 2)  # DeviceToHost
expected_rank = (context.rank - 1) % context.world_size
for i, value in enumerate(result):
    expected = (expected_rank + 1) * 100000 + i * 17 + 3
    if value != expected:
        raise RuntimeError(f"rank={context.rank} remote copy mismatch i={i}: {value} != {expected}")

# Exercise the opposite transfer direction on the real GPU as well.  This
# uses the same public bind/lower/compile path, but its descriptor belongs to
# the remote source rather than the remote destination used by T.copy above.
memcpy(api, dst.data_ptr(), ctypes.addressof(host_zero), COUNT * 4, 1)
if api.hipDeviceSynchronize():
    raise RuntimeError("get destination reset did not synchronize")
context.sync()
get_kernel = tilelang.compile(bind_ipc(dsl_get_block, ["src"]),
                             target="hip -mcpu=gfx936", execution_backend="ctypes")
get_kernel(src, dst, src.device_view(), context.rank, context.world_size)
if api.hipDeviceSynchronize():
    raise RuntimeError("DSL get_block did not synchronize")
context.sync()
memcpy(api, ctypes.addressof(result), dst.data_ptr(), COUNT * 4, 2)
expected_rank = (context.rank + 1) % context.world_size
for i, value in enumerate(result):
    expected = (expected_rank + 1) * 100000 + i * 17 + 3
    if value != expected:
        raise RuntimeError(f"rank={context.rank} get_block mismatch i={i}: {value} != {expected}")

src.close()
dst.close()
context.close()
if mpi.MPI_Finalize():
    raise RuntimeError("MPI_Finalize failed")
print(f"m2-dsl-remote-copy-and-get-ok rank={context.rank}", flush=True)
