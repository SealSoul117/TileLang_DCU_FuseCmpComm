"""Generate the upstream GEMM from old and execute its real HIP entry point.

The development venv has CPU PyTorch. HIP allocations and transfers here avoid
requiring GPU PyTorch, without substituting CPU execution for the HIP kernel.
"""
import ctypes as C
import ctypes.util
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np

from check_paths import check_paths, ROOT


def main():
    check_paths()
    import tilelang

    spec = importlib.util.spec_from_file_location("old_official_gemm", ROOT / "examples/dcu/example_gemm.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    hip = C.CDLL(ctypes.util.find_library("amdhip64") or "libamdhip64.so")
    hip.hipMalloc.argtypes = [C.POINTER(C.c_void_p), C.c_size_t]
    hip.hipFree.argtypes = [C.c_void_p]
    hip.hipMemcpy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]
    hip.hipGetErrorString.argtypes = [C.c_int]
    hip.hipGetErrorString.restype = C.c_char_p

    def checked(error):
        if error:
            raise RuntimeError(hip.hipGetErrorString(error).decode())

    results = []
    for m, n, k in ((512, 768, 256), (1024, 512, 512)):
        func = example.matmul.get_tir(m, n, k, 256, 256, 16)
        kernel = tilelang.compile(func, out_idx=[2], target="hip -mcpu=gfx936",
                                  execution_backend="ctypes", verbose=True,
                                  compile_flags=["--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11"])
        kernel.adapter.lib.get_last_error.restype = C.c_char_p
        if kernel.adapter.lib.init() != 0:
            raise RuntimeError(kernel.adapter.lib.get_last_error().decode())
        artifact = ROOT / "build" / "gemm-smoke" / f"m{m}_n{n}_k{k}"
        artifact.mkdir(parents=True, exist_ok=True)
        (artifact / "kernel.hip").write_text(kernel.get_kernel_source())
        # ctypes keeps the compiled launcher in lib_code; host_kernel_source
        # may be None when engine host code generation is disabled.
        (artifact / "wrapped.hip").write_text(kernel.adapter.lib_code)
        shutil.copy2(kernel.adapter.libpath, artifact / "kernel.so")
        rng = np.random.default_rng(m + n + k)
        a = rng.uniform(-0.5, 0.5, (m, k)).astype(np.float16)
        b = rng.uniform(-0.5, 0.5, (k, n)).astype(np.float16)
        c = np.empty((m, n), dtype=np.float16)
        reference = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)
        pointers = []
        try:
            for array in (a, b, c):
                pointer = C.c_void_p()
                checked(hip.hipMalloc(C.byref(pointer), array.nbytes))
                pointers.append(pointer)
            for array, pointer in zip((a, b), pointers):
                checked(hip.hipMemcpy(pointer, C.c_void_p(array.ctypes.data), array.nbytes, 1))
            launch = kernel.adapter.lib.call
            launch.argtypes = [C.c_void_p] * 4
            launch.restype = C.c_int
            if launch(*pointers, C.c_void_p(0)) != 0:
                raise RuntimeError(kernel.adapter.lib.get_last_error().decode())
            checked(hip.hipGetLastError())
            checked(hip.hipDeviceSynchronize())
            checked(hip.hipMemcpy(C.c_void_p(c.ctypes.data), pointers[2], c.nbytes, 2))
            np.testing.assert_allclose(c, reference, rtol=1e-3, atol=1e-3)
            result = {"shape": [m, n, k], "status": "PASS", "elements": int(c.size),
                      "max_abs_error": float(np.max(np.abs(c.astype(np.float32) - reference))),
                      "jit_library": str(artifact / "kernel.so")}
            results.append(result)
            print(json.dumps(result), flush=True)
        finally:
            for pointer in reversed(pointers):
                checked(hip.hipFree(pointer))
    print(json.dumps({"gemm_smoke": "PASS", "results": results}), flush=True)


if __name__ == "__main__":
    main()
