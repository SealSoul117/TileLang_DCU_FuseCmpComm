"""Pure-GEMM schedule probe for the DCU outer-loop pipeline investigation.

This file deliberately contains no IPC or ReduceScatter code.  It compares
Python/DSL scheduling forms while keeping the DCU GEMM tile fixed at
256x256x16, 512 threads, four pipeline stages, and asynchronous copies.
It is a diagnostic source generator, not a production kernel or a performance
claim.  In particular, only ``flat`` launches one CTA per output tile; the
other probe variants deliberately make one CTA revisit tiles.

Examples (source-only, no GPU access)::

    python python_schedule_probe.py --variant while-hoisted
    python python_schedule_probe.py --all --output generated/python_schedule_probe

An allocated GPU can opt in to compile/run with ``--run``.  The runner uses
HIP ctypes allocations and a ctypes TileLang kernel adapter, so it does not
require PyTorch tensors or MPI.
"""

import argparse
import ctypes
import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np

import tilelang
import tilelang.language as T


TILE_M = 256
TILE_N = 256
BLOCK_K = 16
THREADS = 512
STAGES = 4
TARGET = "hip -mcpu=gfx936"

CUSTOM_VARIANTS = (
    "flat",
    "serial-inner-alloc",
    "serial-hoisted",
    "serial-conservative",
    "persistent-hoisted",
    "while-hoisted",
    "while-inline",
)
REPOSITORY_VARIANT = "repository-persistent"
REPOSITORY_SERIAL_VARIANT = "repository-serial"
ALL_VARIANTS = CUSTOM_VARIANTS + (REPOSITORY_VARIANT, REPOSITORY_SERIAL_VARIANT)


def _check_shape(m: int, n: int, k: int, blocks: int) -> None:
    if min(m, n, k, blocks) <= 0:
        raise ValueError("M, N, K, and blocks must be positive")
    if m % TILE_M or n % TILE_N or k % BLOCK_K:
        raise ValueError("M/N/K must be divisible by 256/256/16 respectively")
    if blocks > (m // TILE_M) * (n // TILE_N):
        raise ValueError("blocks cannot exceed the number of output tiles")


def make_probe(
    variant: str,
    m: int = 1024,
    n: int = 256,
    k: int = 256,
    blocks: int = 1,
    *,
    register_pipeline: bool = True,
    warp_divergence: bool = True,
):
    """Return a pure-GEMM PrimFunc for one Python schedule spelling.

    ``blocks=1`` is the intended persistent diagnostic configuration: a CTA
    visits all four tiles for the default shape.  ``flat`` ignores ``blocks``
    and launches one CTA per tile.  ``while-hoisted`` is the key candidate:
    its outer traversal is a While, not a T.serial For.
    """
    if variant not in CUSTOM_VARIANTS:
        raise ValueError(f"custom variant must be one of {CUSTOM_VARIANTS}, got {variant!r}")
    _check_shape(m, n, k, blocks)
    if warp_divergence and not register_pipeline:
        raise ValueError("warp divergence requires register pipeline; disable both controls together")
    tile_rows, tile_cols = m // TILE_M, n // TILE_N
    tiles, k_blocks = tile_rows * tile_cols, k // BLOCK_K
    if variant != "flat" and tiles % blocks:
        raise ValueError("persistent probe variants require blocks to divide tile count")

    @T.macro
    def emit_gemm_tile(a, b, c, tile_row, tile_col, c_local, a_shared, b_shared):
        """Macro body so the Pipelined loop is parsed as TileLang DSL."""
        T.clear(c_local)
        for kk in T.Pipelined(
            k_blocks,
            num_stages=STAGES,
            enable_register_pipeline=register_pipeline,
            enable_warp_divergence=warp_divergence,
        ):
            T.copy(a[tile_row * TILE_M, kk * BLOCK_K], a_shared,
                   major="row_major", enable_async=True)
            T.copy(b[kk * BLOCK_K, tile_col * TILE_N], b_shared,
                   major="col_major", enable_async=True)
            T.gemm(a_shared, b_shared, c_local, k_pack=1, transpose_C=True)
        T.copy(c_local, c[tile_row * TILE_M, tile_col * TILE_N])
        # Required whenever the same LDS allocation is reused for another tile.
        # Keeping it in flat too makes the end-of-tile protocol uniform.
        T.sync_threads()

    @T.macro
    def emit_conservative_tile(a, b, c, tile_row, tile_col, c_local, a_shared, b_shared):
        """One K step at a time: async layout/copy, wait, GEMM, reuse LDS."""
        T.clear(c_local)
        for kk in T.serial(k_blocks):
            # Keep the original async-copy path and major metadata.  This is
            # deliberately unlike the earlier ``enable_async=False`` attempt,
            # which changes layout lowering in addition to its synchronization.
            T.copy(a[tile_row * TILE_M, kk * BLOCK_K], a_shared,
                   major="row_major", enable_async=True)
            T.copy(b[kk * BLOCK_K, tile_col * TILE_N], b_shared,
                   major="col_major", enable_async=True)
            T.evaluate(T.call_intrin(
                "void", tilelang.tvm.ir.Op.get("tl.async_gld_fence"), 0
            ))
            T.sync_threads()
            T.gemm(a_shared, b_shared, c_local, k_pack=1, transpose_C=True)
            # Do not allow the following copy to reuse LDS while any wave is
            # still consuming this K step.
            T.sync_threads()
        T.copy(c_local, c[tile_row * TILE_M, tile_col * TILE_N])
        T.sync_threads()

    @T.prim_func
    def main(
        a: T.Tensor((m, k), "float16"),
        b: T.Tensor((k, n), "float16"),
        c: T.Tensor((m, n), "float16"),
    ):
        if variant == "flat":
            with T.Kernel(tile_cols, tile_rows, threads=THREADS) as (tile_col, tile_row):
                a_shared = T.alloc_shared((TILE_M, BLOCK_K), "float16")
                b_shared = T.alloc_shared((BLOCK_K, TILE_N), "float16")
                c_local = T.alloc_fragment((TILE_M, TILE_N), "float32")
                emit_gemm_tile(a, b, c, tile_row, tile_col, c_local, a_shared, b_shared)
        elif variant == "serial-inner-alloc":
            with T.Kernel(blocks, threads=THREADS) as block_id:
                # Deliberately mirrors the original problematic spelling:
                # allocations are inside an outer serial tile traversal.
                for tile in T.serial(block_id, tiles, blocks):
                    tile_row, tile_col = tile // tile_cols, tile % tile_cols
                    a_shared = T.alloc_shared((TILE_M, BLOCK_K), "float16")
                    b_shared = T.alloc_shared((BLOCK_K, TILE_N), "float16")
                    c_local = T.alloc_fragment((TILE_M, TILE_N), "float32")
                    emit_gemm_tile(a, b, c, tile_row, tile_col, c_local, a_shared, b_shared)
        elif variant == "serial-hoisted":
            with T.Kernel(blocks, threads=THREADS) as block_id:
                a_shared = T.alloc_shared((TILE_M, BLOCK_K), "float16")
                b_shared = T.alloc_shared((BLOCK_K, TILE_N), "float16")
                c_local = T.alloc_fragment((TILE_M, TILE_N), "float32")
                for tile in T.serial(block_id, tiles, blocks):
                    emit_gemm_tile(a, b, c, tile // tile_cols, tile % tile_cols,
                                   c_local, a_shared, b_shared)
        elif variant == "serial-conservative":
            with T.Kernel(blocks, threads=THREADS) as block_id:
                a_shared = T.alloc_shared((TILE_M, BLOCK_K), "float16")
                b_shared = T.alloc_shared((BLOCK_K, TILE_N), "float16")
                c_local = T.alloc_fragment((TILE_M, TILE_N), "float32")
                for tile in T.serial(block_id, tiles, blocks):
                    emit_conservative_tile(a, b, c, tile // tile_cols, tile % tile_cols,
                                           c_local, a_shared, b_shared)
        elif variant == "persistent-hoisted":
            with T.Kernel(blocks, threads=THREADS) as block_id:
                a_shared = T.alloc_shared((TILE_M, BLOCK_K), "float16")
                b_shared = T.alloc_shared((BLOCK_K, TILE_N), "float16")
                c_local = T.alloc_fragment((TILE_M, TILE_N), "float32")
                # group_size=1 keeps this small-N probe from changing its tile
                # mapping through the usual grouped persistent policy.
                for tile_row, tile_col in T.Persistent(
                    [tile_rows, tile_cols], blocks, block_id, group_size=1
                ):
                    emit_gemm_tile(a, b, c, tile_row, tile_col, c_local, a_shared, b_shared)
        elif variant == "while-hoisted":
            with T.Kernel(blocks, threads=THREADS) as block_id:
                a_shared = T.alloc_shared((TILE_M, BLOCK_K), "float16")
                b_shared = T.alloc_shared((BLOCK_K, TILE_N), "float16")
                c_local = T.alloc_fragment((TILE_M, TILE_N), "float32")
                # alloc_var(init=0) is intentional: unlike a BufferStore based
                # cursor it preserves a shallow surrounding SeqStmt for the
                # DCU prologue pair-barrier insertion pass.
                wave = T.alloc_var("int32", init=0)
                while wave < (tiles + blocks - 1) // blocks:
                    tile = wave * blocks + block_id
                    emit_gemm_tile(a, b, c, tile // tile_cols, tile % tile_cols,
                                   c_local, a_shared, b_shared)
                    wave += 1
        else:  # while-inline
            with T.Kernel(blocks, threads=THREADS) as block_id:
                a_shared = T.alloc_shared((TILE_M, BLOCK_K), "float16")
                b_shared = T.alloc_shared((BLOCK_K, TILE_N), "float16")
                c_local = T.alloc_fragment((TILE_M, TILE_N), "float32")
                wave = T.alloc_var("int32", init=0)
                while wave < (tiles + blocks - 1) // blocks:
                    # Do not call ``emit_gemm_tile`` here: macro arguments are
                    # lowered to loop-local tile_row/tile_col Lets, which the
                    # current DCU resource-descriptor hoist cannot retain.
                    T.clear(c_local)
                    for kk in T.Pipelined(
                        k_blocks,
                        num_stages=STAGES,
                        enable_register_pipeline=register_pipeline,
                        enable_warp_divergence=warp_divergence,
                    ):
                        T.copy(
                            a[((wave * blocks + block_id) // tile_cols) * TILE_M,
                              kk * BLOCK_K],
                            a_shared, major="row_major", enable_async=True,
                        )
                        T.copy(
                            b[kk * BLOCK_K,
                              ((wave * blocks + block_id) % tile_cols) * TILE_N],
                            b_shared, major="col_major", enable_async=True,
                        )
                        T.gemm(a_shared, b_shared, c_local, k_pack=1, transpose_C=True)
                    T.copy(
                        c_local,
                        c[((wave * blocks + block_id) // tile_cols) * TILE_M,
                          ((wave * blocks + block_id) % tile_cols) * TILE_N],
                    )
                    T.sync_threads()
                    wave += 1

    return main


def _repository_persistent(m: int, n: int, k: int, blocks: int, *, primitive: bool):
    """Get the repository persistent example's raw PrimFunc without auto-JIT.

    This is intentionally an *unaligned baseline*: its own 128x256x64,
    256-thread, three-stage schedule is left unchanged.  The temporary module
    patch only replaces ``driver.get_num_sms()`` (which queries CPU Torch) with
    the requested diagnostic CTA count; it does not alter that example's DSL.
    """
    example = Path(__file__).resolve().parents[2] / "gemm" / "example_gemm_persistent.py"
    spec = importlib.util.spec_from_file_location("_schedule_probe_repo_persistent", example)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load repository baseline: {example}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert isinstance(module, ModuleType)
    old_get_num_sms = module.driver.get_num_sms
    module.driver.get_num_sms = lambda: blocks
    try:
        # get_tir invokes the JITImpl's retained original function, avoiding
        # automatic compilation and its original CPU-Torch SM-number query.
        # The original manual serial mapping has group_size=8.  N=256 gives it
        # only one column tile and is therefore not a correctness baseline;
        # use N=2048 (or a multiple) for either repository spelling.
        if n < 2048 or n % 2048:
            raise ValueError("repository baselines require N divisible by 2048 (their original group_size=8 mapping)")
        return module.matmul_persistent.get_tir(
            m, n, k, 128, 256, 64, 256, 3, use_persistent_primitive=primitive
        )
    finally:
        module.driver.get_num_sms = old_get_num_sms


def make_variant(name: str, args: argparse.Namespace):
    if name == REPOSITORY_VARIANT:
        return _repository_persistent(args.m, args.n, args.k, args.blocks, primitive=True)
    if name == REPOSITORY_SERIAL_VARIANT:
        return _repository_persistent(args.m, args.n, args.k, args.blocks, primitive=False)
    return make_probe(name, args.m, args.n, args.k, args.blocks,
                      register_pipeline=not args.no_register_pipeline,
                      warp_divergence=not args.no_warp_divergence)


def export_source(func, out_dir: Path) -> Path:
    """Run lowering only and retain both TIR stages plus emitted HIP source."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # The generic repository GEMM calls target-sensitive layout helpers during
    # lowering.  A Target context is harmless for the DCU-specific variants and
    # lets both kinds of factory use this same source-export entry point.
    import tvm

    with tvm.target.Target(TARGET):
        artifact = tilelang.lower(
            func, target=TARGET, enable_device_compile=False, enable_host_codegen=False
        )
    (out_dir / "raw.tir").write_text(func.script(), encoding="utf-8")
    (out_dir / "device.tir").write_text(artifact.device_mod.script(), encoding="utf-8")
    hip = artifact.kernel_source
    hip_path = out_dir / "gemm.hip"
    hip_path.write_text(hip, encoding="utf-8")
    return hip_path


class _DevicePtr:
    def __init__(self, address: int):
        self.address = address

    def data_ptr(self) -> int:
        return self.address


def _hip():
    api = ctypes.CDLL("libamdhip64.so")
    api.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    api.hipMalloc.restype = ctypes.c_int
    api.hipFree.argtypes = [ctypes.c_void_p]
    api.hipFree.restype = ctypes.c_int
    api.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    api.hipMemcpy.restype = ctypes.c_int
    api.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    api.hipMemset.restype = ctypes.c_int
    api.hipDeviceSynchronize.argtypes = []
    api.hipDeviceSynchronize.restype = ctypes.c_int
    api.hipGetLastError.argtypes = []
    api.hipGetLastError.restype = ctypes.c_int
    api.hipGetErrorString.argtypes = [ctypes.c_int]
    api.hipGetErrorString.restype = ctypes.c_char_p
    return api


def _hip_check(status: int, what: str) -> None:
    if status != 0:
        raise RuntimeError(f"{what} failed with HIP status {status}")


def _adapter_error(lib) -> str:
    """Read the wrapper's fixed error buffer after an init/call failure."""
    lib.get_last_error.argtypes = []
    lib.get_last_error.restype = ctypes.c_char_p
    message = lib.get_last_error()
    return message.decode("utf-8", errors="replace") if message else "(no wrapper error text)"


def _checked_adapter_init(kernel) -> None:
    """Compensate for the current ctypes adapter discarding ``init`` status."""
    lib = kernel.adapter.lib
    lib.init.argtypes = []
    lib.init.restype = ctypes.c_int
    status = lib.init()
    if status:
        raise RuntimeError(f"ctypes wrapper init failed ({status}): {_adapter_error(lib)}")


def _checked_ctypes_call(kernel, api, a: _DevicePtr, b: _DevicePtr, c: _DevicePtr) -> None:
    """Use the three-pointer wrapper ABI directly and retain its status."""
    lib = kernel.adapter.lib
    lib.call.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.call.restype = ctypes.c_int  # PREDEF_HOST_FUNC declares ``extern C int call(...)``.
    status = lib.call(ctypes.c_void_p(a.address), ctypes.c_void_p(b.address),
                      ctypes.c_void_p(c.address), ctypes.c_void_p(0))
    if status:
        raise RuntimeError(f"ctypes wrapper call failed ({status}): {_adapter_error(lib)}")
    hip_status = api.hipGetLastError()
    if hip_status:
        text = api.hipGetErrorString(hip_status)
        detail = text.decode("utf-8", errors="replace") if text else "unknown HIP error"
        raise RuntimeError(f"hipGetLastError after wrapper call failed ({hip_status}): {detail}")


def _device_alloc(api, nbytes: int) -> _DevicePtr:
    ptr = ctypes.c_void_p()
    _hip_check(api.hipMalloc(ctypes.byref(ptr), nbytes), "hipMalloc")
    return _DevicePtr(ptr.value)


def _copy_h2d(api, dst: _DevicePtr, array: np.ndarray) -> None:
    _hip_check(api.hipMemcpy(ctypes.c_void_p(dst.address), ctypes.c_void_p(array.ctypes.data),
                             array.nbytes, 1), "hipMemcpy H2D")


def _copy_d2h(api, dst: np.ndarray, src: _DevicePtr) -> None:
    _hip_check(api.hipMemcpy(ctypes.c_void_p(dst.ctypes.data), ctypes.c_void_p(src.address),
                             dst.nbytes, 2), "hipMemcpy D2H")


def _inputs(m: int, n: int, k: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20261001)
    if mode == "dyadic":
        a = (rng.integers(-8, 9, (m, k), dtype=np.int16).astype(np.float32) / 16).astype(np.float16)
        b = (rng.integers(-8, 9, (k, n), dtype=np.int16).astype(np.float32) / 16).astype(np.float16)
    else:
        a = rng.uniform(-0.25, 0.25, (m, k)).astype(np.float16)
        b = rng.uniform(-0.25, 0.25, (k, n)).astype(np.float16)
    return a, b


def run_one(func, args: argparse.Namespace, kernel=None) -> None:
    """Compile and execute one locally allocated HIP GEMM, then check C."""
    a, b = _inputs(args.m, args.n, args.k, args.input_mode)
    reference = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)
    output = np.empty_like(reference)
    if kernel is None:
        kernel = _compile(func)
    _checked_adapter_init(kernel)
    api = _hip()
    a_dev = b_dev = c_dev = None
    try:
        a_dev, b_dev, c_dev = (_device_alloc(api, x.nbytes) for x in (a, b, output))
        _copy_h2d(api, a_dev, a)
        _copy_h2d(api, b_dev, b)
        _hip_check(api.hipMemset(ctypes.c_void_p(c_dev.address), 0, output.nbytes), "hipMemset")
        _checked_ctypes_call(kernel, api, a_dev, b_dev, c_dev)
        _hip_check(api.hipDeviceSynchronize(), "kernel synchronize")
        _copy_d2h(api, output, c_dev)
    finally:
        for allocation in (a_dev, b_dev, c_dev):
            if allocation is not None:
                api.hipFree(ctypes.c_void_p(allocation.address))
    max_abs = float(np.max(np.abs(output.astype(np.float32) - reference.astype(np.float32))))
    nan_count = int(np.count_nonzero(np.isnan(output)))
    inf_count = int(np.count_nonzero(np.isinf(output)))
    if args.input_mode == "dyadic":
        mismatches = int(np.count_nonzero(output.view(np.uint16) != reference.view(np.uint16)))
        print(f"dyadic: max_abs={max_abs:g}, nan={nan_count}, inf={inf_count}, bit_mismatches={mismatches}")
        if mismatches:
            raise AssertionError("dyadic result is not bitwise equal to FP32-accumulate reference")
    else:
        mismatches = int(np.count_nonzero(~np.isclose(output, reference, rtol=1e-3, atol=1e-3)))
        print(f"continuous: max_abs={max_abs:g}, nan={nan_count}, inf={inf_count}, mismatches(rtol=atol=1e-3)={mismatches}")
        if mismatches:
            raise AssertionError("continuous result exceeds rtol=atol=1e-3")


def _compile(func):
    """Compile in the same explicit Target context used for source export."""
    import tvm

    tilelang.disable_cache()
    with tvm.target.Target(TARGET):
        return tilelang.compile(func, out_idx=[], target=TARGET, execution_backend="ctypes")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=ALL_VARIANTS, default="while-hoisted")
    parser.add_argument("--all", action="store_true",
                        help="source-export the five comparable 256x256 DCU variants")
    parser.add_argument("--M", dest="m", type=int, default=1024)
    parser.add_argument("--N", dest="n", type=int, default=256,
                        help="N=2048 (or a multiple) is required for repository-* baselines")
    parser.add_argument("--K", dest="k", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=1, help="persistent CTA count; flat ignores it")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "generated" / "python_schedule_probe")
    parser.add_argument("--compile", action="store_true", help="also invoke TileLang HIP compilation")
    parser.add_argument("--run", action="store_true", help="compile, launch once, and validate one variant")
    parser.add_argument("--input-mode", choices=("dyadic", "continuous"), default="dyadic")
    parser.add_argument("--no-register-pipeline", action="store_true")
    parser.add_argument("--no-warp-divergence", action="store_true")
    args = parser.parse_args()
    if args.all and (args.compile or args.run):
        parser.error("--all is source-export only; choose one variant for --compile/--run")
    if args.run:
        args.compile = True
    _check_shape(args.m, args.n, args.k, args.blocks)
    return args


def main() -> None:
    args = _parse_args()
    # The repository baselines intentionally retain their own 128x256x64,
    # group-size-eight schedule and need N>=2048; export them explicitly
    # rather than making ordinary ``--all`` with the N=256 DCU shape fail.
    names = CUSTOM_VARIANTS if args.all else (args.variant,)
    for name in names:
        func = make_variant(name, args)
        hip_path = export_source(func, args.output / name)
        print(f"[{name}] lowered HIP: {hip_path}")
        compiled = None
        if args.compile:
            compiled = _compile(func)
            (args.output / name / "compiled.hip").write_text(compiled.get_kernel_source(), encoding="utf-8")
            print(f"[{name}] compiled HIP: {args.output / name / 'compiled.hip'}")
        if args.run:
            run_one(func, args, compiled)


if __name__ == "__main__":
    main()
