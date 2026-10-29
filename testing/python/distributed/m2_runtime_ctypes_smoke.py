"""Four-process smoke for the lazy Python wrapper of the embedded M2 runtime.

Run under mpirun; this test deliberately owns MPI init/finalize in the runner,
as required by the public C ABI.
"""
import ctypes


mpi = ctypes.CDLL("libmpi.so")
provided = ctypes.c_int()
if mpi.MPI_Init_thread(None, None, 1, ctypes.byref(provided)):
    raise RuntimeError("MPI_Init_thread failed")

from tilelang.distributed.ipc_runtime import IPCContext

context = IPCContext()
payload = context.alloc("python-ctypes-smoke", 4096)
assert payload.data_ptr() and payload.device_view().data_ptr()
context.sync()
payload.close()
context.close()
if mpi.MPI_Finalize():
    raise RuntimeError("MPI_Finalize failed")
print("m2-python-ctypes-runtime-ok", flush=True)
