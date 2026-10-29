"""Per-rank HIP-event benchmark for the bounded TileLang GEMM+AllReduce.

All modes use the same TileLang IPC allocations: A/B DeviceDefault and
partial/ready/out FineGrained.  ``compare`` alternates TileLang and the
provided fused-vector C-ABI kernel without changing allocations or inputs.
Run with four MPI ranks on one DCU node.
"""
from __future__ import annotations

import argparse
import ctypes
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
RS = HERE.parent / "gemm_rs_prototype"
if str(RS) not in sys.path:
    sys.path.insert(0, str(RS))

import run_gemm_reduce_scatter as runner
from benchmark_gemm_reduce_scatter import (EventPair, TileLangRun as RsTileLangRun,
                                           _configure_hip, _mpi_abort)
from gemm_reduce_scatter_conservative import _check_adapter_init
from gemm_allreduce_scheduled import (BLOCK_K, DEFAULT_TOTAL_BLOCKS, PE_COUNT, THREADS,
                                      TILE_M, TILE_N, _check_shape, build_bound,
                                      default_comm_blocks, schedule_parameters)

TARGET = "hip -mcpu=gfx936"
HOST_TO_DEVICE = runner.HOST_TO_DEVICE
DEVICE_TO_HOST = runner.DEVICE_TO_HOST
DEVICE_VIEW_WORDS = 6  # reference C ABI explicitly takes a host uint64[6] copy


def _verify_tilelang_ar_vector_source(source: str, enabled: bool) -> None:
    """Reject a benchmark binary whose JIT lowering missed the requested pass."""
    marker = "ipc_ar_half8_i"
    if not enabled:
        if marker in source:
            raise RuntimeError("--scalar requested but JIT HIP contains the AR half8 loop")
        return
    pull_lines = [line for line in source.splitlines()
                  if "*(uint4*)" in line and "remote_partial" in line]
    push_lines = [line for line in source.splitlines()
                  if "*(uint4*)" in line and "remote_out" in line]
    if marker not in source or len(pull_lines) != 4 or len(push_lines) != 4:
        raise RuntimeError("TileLang IPC half8 pass was requested but JIT HIP did not emit 4 uint4 pulls/stores")
    for peer in range(4):
        if sum(f"remote_partial{peer}_remote_ptr_" in line for line in pull_lines) != 1 or \
           sum(f"remote_out{peer}_remote_ptr_" in line for line in push_lines) != 1:
            raise RuntimeError("TileLang IPC half8 JIT HIP is missing a peer-specific uint4 transfer")
    if source.count("uint4 ipc_ar_reduced_f16x8 =") != 1 or \
       not all("= ipc_ar_reduced_f16x8;" in line for line in push_lines):
        raise RuntimeError("TileLang IPC half8 JIT HIP did not bind one shared reduced vector")


def _rank_from_environment() -> int:
    for name in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "MV2_COMM_WORLD_RANK"):
        if name in os.environ:
            return int(os.environ[name])
    raise RuntimeError("could not determine MPI rank")


def _abort(mpi, reason: str) -> None:
    print(f"GEMM+AllReduce abort: {reason}", flush=True)
    _mpi_abort(mpi)


@lru_cache(maxsize=1)
def _expected(rank: int, m: int, n: int, k: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Cache immutable CPU expectations, never copied device results."""
    partial = runner._tile_view(runner._partial_reference(rank, m, n, k, mode))
    # Keep the reference kernel's PE0, PE1, PE2, PE3 addition order.
    reduced = (runner._partial_reference(0, m, n, k, mode).astype(np.float32)
               + runner._partial_reference(1, m, n, k, mode).astype(np.float32)
               + runner._partial_reference(2, m, n, k, mode).astype(np.float32)
               + runner._partial_reference(3, m, n, k, mode).astype(np.float32)).astype(np.float16)
    return partial, reduced


def _full_check(hip, partial_ptr: int, out_ptr: int, rank: int, m: int, n: int, k: int,
                mode: str) -> dict[str, object]:
    """Check local tile-major partial and this PE's complete allreduce output."""
    tiles = (m // TILE_M) * (n // TILE_N)
    partial = np.empty((tiles, TILE_M, TILE_N), dtype=np.float16)
    out = np.empty((m, n), dtype=np.float16)
    runner._copy(hip, partial.ctypes.data, partial_ptr, partial.nbytes, DEVICE_TO_HOST)
    runner._copy(hip, out.ctypes.data, out_ptr, out.nbytes, DEVICE_TO_HOST)
    runner._check(hip.hipDeviceSynchronize(), "allreduce result copies")
    expected_partial, reduced = _expected(rank, m, n, k, mode)
    runner._check_result(partial, expected_partial, "partial", mode, rank)
    runner._check_result(out, reduced, "allreduce", mode, rank)
    return {"mode": f"full-{mode}", "status": "passed"}


class Run:
    """One IPC allocation set, with TileLang and/or reference launchers."""
    # Reuse the reviewed occupancy query implementation from the sibling
    # bounded prototype; its helper only needs the attributes initialized here.
    _require_full_residency_function = RsTileLangRun._require_full_residency_function

    def __init__(self, args):
        from tilelang.distributed.ipc_runtime import IPCContext

        self.args = args
        self.mpi = ctypes.CDLL("libmpi.so")
        provided = ctypes.c_int()
        runner._check(self.mpi.MPI_Init_thread(None, None, 1, ctypes.byref(provided)), "MPI_Init_thread")
        self.ctx = IPCContext()
        self.rank = self.ctx.rank
        if self.ctx.world_size != PE_COUNT:
            raise RuntimeError(f"requires {PE_COUNT} MPI ranks, got {self.ctx.world_size}")
        self.hip = _configure_hip()
        self.schedule = schedule_parameters(args.m, args.n, args.total_blocks, args.comm_blocks, k=args.k)
        self.tile_count = self.schedule["tile_count"]
        self.flag_words = self.schedule["flag_words"]
        self.allocations = [
            self.ctx.alloc("gemm-ar-a", args.m * args.k * 2, fine_grained=False),
            self.ctx.alloc("gemm-ar-b", args.k * args.n * 2, fine_grained=False),
            self.ctx.alloc("gemm-ar-partial", self.tile_count * TILE_M * TILE_N * 2, fine_grained=True),
            self.ctx.alloc("gemm-ar-ready", self.flag_words * 4, fine_grained=True),
            self.ctx.alloc("gemm-ar-out", args.m * args.n * 2, fine_grained=True),
        ]
        self.a, self.b, self.partial, self.ready, self.out = self.allocations
        self.launched = False
        self.reader_safe = False
        self.residency = None
        self.residencies = {}
        self._initialize_inputs()
        self.tile_kernel = self._compile_tilelang() if args.implementation in ("tilelang", "compare") else None
        self.reference = self._open_reference() if args.implementation in ("handwritten", "compare") else None
        # The reference deliberately takes host copies of the opaque runtime
        # descriptors.  Copying occurs before events and the arrays stay alive
        # for every C-ABI launch.
        self.partial_view_words = self._copy_host_view(self.partial)
        self.out_view_words = self._copy_host_view(self.out)
        self.ready_view_words = self._copy_host_view(self.ready)
        self._tile_args = self._prebind_tile_args() if self.tile_kernel is not None else None
        self._reference_args = self._prebind_reference_args() if self.reference is not None else None

    def _initialize_inputs(self) -> None:
        a, b = runner._rank_inputs(self.rank, self.args.m, self.args.n, self.args.k, self.args.input_mode)
        runner._copy(self.hip, self.a.data_ptr(), a.ctypes.data, a.nbytes, HOST_TO_DEVICE)
        runner._copy(self.hip, self.b.data_ptr(), b.ctypes.data, b.nbytes, HOST_TO_DEVICE)
        runner._check(self.hip.hipMemset(ctypes.c_void_p(self.ready.data_ptr()), 0, self.flag_words * 4), "initial ready reset")
        runner._check(self.hip.hipDeviceSynchronize(), "input initialization")
        self.ctx.sync()

    def _compile_tilelang(self):
        import tilelang
        tilelang.disable_cache()
        _, bound = build_bound(self.args.m, self.args.n, self.args.k, total_blocks=self.args.total_blocks,
                              comm_blocks=self.args.comm_blocks)
        # JITKernel establishes its own PassContext from pass_configs, so an
        # outer context here would not reach the lowering that produces HIP.
        kernel = tilelang.compile(
            bound, target=TARGET, execution_backend="ctypes",
            pass_configs={"tl.enable_dcu_ipc_vectorize": not self.args.scalar},
        )
        _check_adapter_init(kernel)
        kernel_source = kernel.get_kernel_source()
        _verify_tilelang_ar_vector_source(kernel_source, enabled=not self.args.scalar)
        if self.args.export_hip:
            self.args.export_hip.mkdir(parents=True, exist_ok=True)
            (self.args.export_hip / f"tilelang-ar-m{self.args.m}-n{self.args.n}-k{self.args.k}-rank{self.rank}.hip").write_text(
                kernel_source, encoding="utf-8")
        # Reuse the established 80 CTA / 512 thread / 64 KiB liveness check.
        RsTileLangRun._require_full_residency(self, kernel)
        self.residencies["tilelang"] = dict(self.residency)
        return kernel

    def _open_reference(self):
        path = self.args.reference_library
        if path is None:
            path = HERE.parents[2] / "build-old" / "gemm-ar-20261011" / f"reference-m{self.args.m}.so"
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        lib.gemm_ar_reference_shape.argtypes = [ctypes.POINTER(ctypes.c_int)] * 3
        lib.gemm_ar_reference_shape.restype = ctypes.c_int
        sm, sn, sk = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        if lib.gemm_ar_reference_shape(ctypes.byref(sm), ctypes.byref(sn), ctypes.byref(sk)) or (sm.value, sn.value, sk.value) != (self.args.m, self.args.n, self.args.k):
            raise RuntimeError(f"reference library shape {(sm.value, sn.value, sk.value)} does not match CLI")
        lib.gemm_ar_reference_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.gemm_ar_reference_launch.restype = ctypes.c_int
        lib.gemm_ar_reference_kernel.argtypes = []
        lib.gemm_ar_reference_kernel.restype = ctypes.c_void_p
        function = lib.gemm_ar_reference_kernel()
        if not function:
            raise RuntimeError("reference returned a null kernel function")
        RsTileLangRun._require_full_residency_function(self, ctypes.c_void_p(function))
        self.residencies["handwritten"] = dict(self.residency)
        return lib

    def _copy_host_view(self, allocation):
        words = (ctypes.c_uint64 * DEVICE_VIEW_WORDS)()
        runner._copy(self.hip, ctypes.addressof(words), allocation.device_view().data_ptr(), ctypes.sizeof(words), DEVICE_TO_HOST)
        runner._check(self.hip.hipDeviceSynchronize(), "copy IPC DeviceView to host")
        return words

    def _prebind_tile_args(self):
        """Derive/order the C wrapper payload from adapter metadata once."""
        functions = [func for func in self.tile_kernel.adapter.ir_module.functions.values()
                     if hasattr(func, "params") and hasattr(func, "buffer_map")]
        if len(functions) != 1:
            raise RuntimeError(f"expected one adapter PrimFunc, found {len(functions)}")
        prim = functions[0]
        def param_name(param):
            buffer = prim.buffer_map.get(param)
            if buffer is not None:
                return str(buffer.name)
            # M1 exposes Var.name; newer TVM exposes name_hint.  The
            # descriptor/scalar ABI check must work with either binding.
            return str(getattr(param, "name", getattr(param, "name_hint", "")))
        names = [param_name(param) for param in prim.params]
        if len(names) != 10:
            raise RuntimeError(f"expected ten bound TileLang ABI parameters, got {len(names)}: {names}")
        # prepare_ipc_primfunc retains the original payload parameters, then
        # appends the bound descriptors and finally the two scalar arguments.
        # This check makes the prebound ctypes argument order evidence-based.
        expected = ("A", "B", "partial", "ready", "out", "ipc_view_partial",
                    "ipc_view_ready", "ipc_view_out", "__tl_ipc_rank", "__tl_ipc_world_size")
        if tuple(names) != expected:
            raise RuntimeError(f"unexpected TileLang C-ABI parameter order: {names}; expected {expected}")
        lib = self.tile_kernel.adapter.lib
        lib.call.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p]
        lib.call.restype = ctypes.c_int
        return (ctypes.c_void_p(self.a.data_ptr()), ctypes.c_void_p(self.b.data_ptr()),
                ctypes.c_void_p(self.partial.data_ptr()), ctypes.c_void_p(self.ready.data_ptr()),
                ctypes.c_void_p(self.out.data_ptr()), ctypes.c_void_p(self.partial.device_view().data_ptr()),
                ctypes.c_void_p(self.ready.device_view().data_ptr()), ctypes.c_void_p(self.out.device_view().data_ptr()),
                ctypes.c_int32(self.rank), ctypes.c_int32(self.ctx.world_size), ctypes.c_void_p(0))

    def _prebind_reference_args(self):
        """Freeze the complete handwritten C ABI payload before timing."""
        return (ctypes.c_void_p(self.a.data_ptr()), ctypes.c_void_p(self.b.data_ptr()),
                self.partial_view_words, self.out_view_words, self.ready_view_words,
                ctypes.c_int(self.rank), ctypes.c_int(self.args.total_blocks),
                ctypes.c_int(self.schedule["compute_blocks"]), ctypes.c_void_p(0))

    def prepare(self) -> None:
        runner._check(self.hip.hipMemset(ctypes.c_void_p(self.ready.data_ptr()), 0, self.flag_words * 4), "ready reset")
        runner._check(self.hip.hipDeviceSynchronize(), "ready reset completion")
        self.ctx.sync()

    def poison_results(self) -> None:
        """Make a missed partial/output write visible in an untimed oracle run."""
        runner._check(self.hip.hipMemset(ctypes.c_void_p(self.partial.data_ptr()), 0xA5,
                                         self.tile_count * TILE_M * TILE_N * 2), "partial poison")
        runner._check(self.hip.hipMemset(ctypes.c_void_p(self.out.data_ptr()), 0xA5,
                                         self.args.m * self.args.n * 2), "out poison")
        runner._check(self.hip.hipDeviceSynchronize(), "poison completion")
        self.ctx.sync()

    def launch(self, which: str) -> None:
        self.launched, self.reader_safe = True, False
        if which == "tilelang":
            assert self.tile_kernel is not None and self._tile_args is not None
            status = self.tile_kernel.adapter.lib.call(*self._tile_args)
            if status:
                raise RuntimeError(f"TileLang C-ABI launch returned {status}")
        elif which == "handwritten":
            assert self.reference is not None and self._reference_args is not None
            status = self.reference.gemm_ar_reference_launch(*self._reference_args)
            if status:
                raise RuntimeError(f"handwritten C-ABI launch returned {status}")
        else:
            raise ValueError(which)
        code = self.hip.hipGetLastError()
        if code:
            raise RuntimeError(f"launch HIP error {code}")

    def complete(self) -> None:
        # First only local HIP sync, proving the in-kernel peer handshake is
        # sufficient for a complete local full-output check.  The context
        # synchronization then supplies the safe next-round boundary.
        runner._check(self.hip.hipDeviceSynchronize(), "post-kernel local synchronize")
        self.reader_safe = True

    def finish_round(self) -> None:
        self.ctx.sync()

    @property
    def partial_ptr(self): return self.partial.data_ptr()
    @property
    def out_ptr(self): return self.out.data_ptr()

    def close(self) -> None:
        if self.launched and not self.reader_safe:
            os._exit(1)
        for allocation in reversed(self.allocations): allocation.close()
        self.ctx.close()
        runner._check(self.mpi.MPI_Finalize(), "MPI_Finalize")


def _one(state: Run, events: EventPair, which: str, timeout: float) -> float:
    state.prepare()
    elapsed = events.time_one(lambda: state.launch(which), timeout)
    state.complete()
    return elapsed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--implementation", choices=("tilelang", "handwritten", "compare"), default="tilelang")
    p.add_argument("--reference-library", type=Path)
    p.add_argument("--m", type=int, default=2048); p.add_argument("--n", type=int, default=2048); p.add_argument("--k", type=int, default=512)
    p.add_argument("--total-blocks", type=int, default=DEFAULT_TOTAL_BLOCKS); p.add_argument("--comm-blocks", type=int)
    p.add_argument("--scalar", action="store_true", help="disable the TileLang IPC half8 pass"); p.add_argument("--input-mode", choices=("dyadic", "continuous"), default="dyadic")
    p.add_argument("--warmup", type=int, default=5); p.add_argument("--repeats", type=int, default=30); p.add_argument("--timeout-seconds", type=float, default=60)
    p.add_argument("--json-dir", type=Path, default=HERE / "benchmark_results"); p.add_argument("--export-hip", type=Path)
    args = p.parse_args()
    _check_shape(args.m, args.n, args.k)
    if args.comm_blocks is None: args.comm_blocks = default_comm_blocks(args.m, args.n, args.k)
    if args.warmup < 0 or args.repeats <= 0 or args.timeout_seconds <= 0: raise ValueError("invalid warmup/repeats/timeout")
    state = None
    events = None
    try:
        state = Run(args)
        events = EventPair(state.hip)
        print(f"rank={state.rank} allreduce precheck grid={state.schedule} residency={state.residencies}", flush=True)
        variants = ("tilelang", "handwritten") if args.implementation == "compare" else (args.implementation,)
        # A local sync happens before this check, then the explicit cross-PE
        # barrier. This catches a missing in-kernel completion protocol.
        pre = {}
        for v in variants:
            state.poison_results()
            _one(state, events, v, args.timeout_seconds)
            pre[v] = _full_check(state.hip, state.partial_ptr, state.out_ptr, state.rank, args.m, args.n, args.k, args.input_mode)
            state.finish_round()
        for v in variants:
            for _ in range(args.warmup):
                _one(state, events, v, args.timeout_seconds); state.finish_round()
        samples = {v: [] for v in variants}
        for i in range(args.repeats):
            # Alternating makes compare share the same reset/input lifecycle.
            order = variants if i % 2 == 0 else tuple(reversed(variants))
            for v in order:
                samples[v].append(_one(state, events, v, args.timeout_seconds)); state.finish_round()
        post = {}
        for v in variants:
            state.poison_results()
            _one(state, events, v, args.timeout_seconds)
            post[v] = _full_check(state.hip, state.partial_ptr, state.out_ptr, state.rank, args.m, args.n, args.k, args.input_mode)
            state.finish_round()
    except BaseException as error:
        _abort(state.mpi if state is not None else ctypes.CDLL("libmpi.so"), str(error))
    finally:
        if events is not None:
            events.close()
        if state is not None:
            state.close()
    args.json_dir.mkdir(parents=True, exist_ok=True)
    output = args.json_dir / f"gemm-ar-{args.implementation}-m{args.m}-n{args.n}-k{args.k}-rank{state.rank}.json"
    doc = {"schema": "gemm-ar-per-pe-hip-event-v1", "implementation": args.implementation,
           "shape": {"m": args.m, "n": args.n, "k": args.k, "pe_count": PE_COUNT}, "rank": state.rank,
           "grid": {"total_blocks": args.total_blocks, "comm_blocks": args.comm_blocks, "compute_blocks": state.schedule["compute_blocks"]},
           "flags": {"words": state.flag_words, "tile_ready_words": state.schedule["tile_ready_words"], "peer_done_words": 4, "block_done_words": args.comm_blocks},
           "allocations": "A/B DeviceDefault; partial/ready/out FineGrained", "residency": state.residency,
           "input_mode": args.input_mode, "warmup": args.warmup, "repeats": args.repeats,
           "timing": {v: {"samples_ms": values, "median_ms": float(statistics.median(values))} for v, values in samples.items()},
           "verification": {"pre_timing": pre, "post_timing": post}, "reference_library": str(args.reference_library) if args.reference_library else None,
           "notes": ["samples are raw per-rank HIP event durations; aggregate ranks with elementwise max", "TileLang uses a scalar AR frontend; --scalar disables the verified IPC half8 pass. Inspect exported HIP for actual vector instructions."]}
    output.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"gemm-ar benchmark rank={state.rank} json={output}", flush=True)


if __name__ == "__main__": main()
