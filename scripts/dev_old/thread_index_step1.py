"""Real DSL -> HIP -> device checks for raw and asm-opaque thread indices.

Run in an allocated gfx936 node with scripts/dev_old/env.sh. The existing
max256/launch512 convention is deliberately retained to exercise the failure.
"""
import argparse
import ctypes as C
import json
from pathlib import Path

import numpy as np
import tilelang
import tilelang.language as T


def make_probe(n, wide, opaque=False):
    @T.prim_func
    def probe(out: T.Tensor((2, n), "int32")):
        with T.Kernel(2, threads=512) as bx:
            if opaque:
                T.import_source("#include <tl_templates/dcu_hip/ipc.hpp>\n")
            tx = T.get_thread_binding()
            comm_tx = T.call_extern("int32", "tl::ipc::opaque_thread_id", tx) if opaque else tx
            for linear in T.serial(comm_tx, n, 512):
                if wide:
                    out[bx, T.cast(linear, "int64")] = linear + comm_tx * 100000 + bx * 100000000
                else:
                    out[bx, linear] = linear + comm_tx * 100000 + bx * 100000000
    return probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--opaque", action="store_true", help="Use the production asm-opaque helper")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tilelang.disable_cache()
    hip = C.CDLL("libamdhip64.so")
    hip.hipMalloc.argtypes = [C.POINTER(C.c_void_p), C.c_size_t]
    hip.hipMemset.argtypes = [C.c_void_p, C.c_int, C.c_size_t]
    hip.hipMemcpy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]
    hip.hipFree.argtypes = [C.c_void_p]
    hip.hipGetErrorString.argtypes = [C.c_int]
    hip.hipGetErrorString.restype = C.c_char_p

    def check(code):
        if code:
            raise RuntimeError(hip.hipGetErrorString(code).decode())

    reports = []
    for n in (65536, 1003):
        for wide in (False, True):
            name = f"n{n}-{'int64' if wide else 'int32'}"
            func = make_probe(n, wide, args.opaque)
            with tilelang.tvm.target.Target("hip -mcpu=gfx936"):
                artifact = tilelang.lower(func, target="hip -mcpu=gfx936",
                                          enable_device_compile=False, enable_host_codegen=False)
            (args.output / f"{name}.hip").write_text(artifact.kernel_source)
            (args.output / f"{name}.tir").write_text(artifact.device_mod.script())
            if args.source_only:
                continue
            kernel = tilelang.compile(func, target="hip -mcpu=gfx936", execution_backend="ctypes",
                                      compile_flags=["--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11"])
            lib = kernel.adapter.lib
            lib.get_last_error.restype = C.c_char_p
            if lib.init() != 0:
                raise RuntimeError(lib.get_last_error().decode())
            call = lib.call
            call.argtypes = [C.c_void_p, C.c_void_p]
            call.restype = C.c_int
            # A padded sentinel region also catches a final block writing past
            # the tensor boundary when a tail bound is miscompiled.
            storage = np.empty(2 * n + 512, dtype=np.int32)
            result = storage[:2 * n].reshape(2, n)
            index = np.arange(n, dtype=np.int32)
            expected = index[None, :] + (index % 512)[None, :] * 100000 + np.arange(2, dtype=np.int32)[:, None] * 100000000
            device = C.c_void_p()
            check(hip.hipMalloc(C.byref(device), storage.nbytes))
            try:
                check(hip.hipMemset(device, 0xff, storage.nbytes))
                if call(device, C.c_void_p(0)) != 0:
                    raise RuntimeError(lib.get_last_error().decode())
                check(hip.hipDeviceSynchronize())
                check(hip.hipMemcpy(C.c_void_p(storage.ctypes.data), device, storage.nbytes, 2))
                mismatch = int(np.count_nonzero(result != expected))
                guard_corruptions = int(np.count_nonzero(storage[2 * n:] != -1))
                report = {"case": name, "elements": result.size, "mismatches": mismatch,
                          "index_mode": "asm-opaque" if args.opaque else "raw",
                          "untouched": int(np.count_nonzero(result == -1)),
                          "guard_corruptions": guard_corruptions,
                          "status": "PASS" if mismatch == 0 and guard_corruptions == 0 else "FAIL"}
                reports.append(report)
                print(json.dumps(report), flush=True)
            finally:
                check(hip.hipFree(device))
    if not args.source_only:
        (args.output / "results.json").write_text(json.dumps(reports, indent=2) + "\n")
        assert all(r["status"] == "PASS" for r in reports), reports


if __name__ == "__main__":
    main()
