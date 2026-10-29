"""Run the original four-stage GEMM+RS with checked ctypes initialization/launch.

This adds host error checks only: the PrimFunc, IPC protocol, inputs and numerical
assertions remain those of run_gemm_reduce_scatter.py. Run under four MPI ranks.
"""
import ctypes
import os
from pathlib import Path

import run_gemm_reduce_scatter as runner
from gemm_reduce_scatter_conservative import _check_adapter_init, _checked_launch


_original_compile = runner.tilelang.compile


class CheckedKernel:
    def __init__(self, kernel):
        self.kernel = kernel
        _check_adapter_init(kernel)
        rank = os.environ.get("OMPI_COMM_WORLD_RANK", "unknown")
        directory = Path(os.environ.get("GEMM_RS_ARTIFACT_DIR", "build-old/pipeline-fix/fused"))
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"rank{rank}.hip").write_text(kernel.get_kernel_source(), encoding="utf-8")
        self.hip = runner._hip()
        self.hip.hipGetLastError.argtypes = []
        self.hip.hipGetLastError.restype = ctypes.c_int
        self.hip.hipGetErrorString.argtypes = [ctypes.c_int]
        self.hip.hipGetErrorString.restype = ctypes.c_char_p

    def __call__(self, *args):
        try:
            _checked_launch(self.kernel, self.hip, *args)
        except BaseException as error:
            # The wrapper may have submitted work before reporting an error.
            # Never collectively free potentially live remote allocations here.
            print(f"checked launch failed: {error}", flush=True)
            os._exit(1)


def checked_compile(*args, **kwargs):
    return CheckedKernel(_original_compile(*args, **kwargs))


if __name__ == "__main__":
    runner.tilelang.compile = checked_compile
    runner.main()
