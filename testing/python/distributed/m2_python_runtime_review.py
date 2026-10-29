"""Independent four-PE check of the lazy Python runtime's lifetime checks."""
import ctypes
import threading

from tilelang.distributed import IPCContext, IPCError


mpi = ctypes.CDLL("libmpi.so")
provided = ctypes.c_int()
assert mpi.MPI_Init_thread(None, None, 1, ctypes.byref(provided)) == 0
assert provided.value >= 1
context = IPCContext()
assert context.world_size == 4
allocation = context.alloc("python-lifetime-review", 4096)
view = allocation.device_view()
address = view.data_ptr()
errors = []


def wrong_thread():
    try:
        allocation.device_view()
    except IPCError:
        errors.append("rejected")


worker = threading.Thread(target=wrong_thread)
worker.start()
worker.join()
assert errors == ["rejected"], "Python cache bypassed C ABI owner-thread check"
assert allocation.device_view().data_ptr() == address
allocation.close()
assert allocation.closed
for invalid_access in (allocation.data_ptr, allocation.device_view, view.data_ptr):
    try:
        invalid_access()
    except IPCError:
        pass
    else:
        raise AssertionError("freed allocation/descriptor remained accessible")
context.close()
finalized = ctypes.c_int()
assert mpi.MPI_Finalized(ctypes.byref(finalized)) == 0 and finalized.value == 0
print(f"PASS Python lifetime rank={context.rank}", flush=True)
assert mpi.MPI_Finalize() == 0
