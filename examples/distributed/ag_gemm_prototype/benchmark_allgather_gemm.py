"""Four-rank HIP-event runner for the bounded TileLang AllGather + GEMM prototype.

The benchmark is TileLang-only: it intentionally measures the scalar frontend
or the opt-in verified IPC half8 pass with identical IPC allocations.  Every
rank writes raw event samples; aggregate an experiment with the elementwise
maximum across ranks before taking a median.
"""

import argparse
import ctypes
import json
import os
from pathlib import Path
import statistics
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
RS = HERE.parent / "gemm_rs_prototype"
if str(RS) not in sys.path:
    sys.path.insert(0, str(RS))

import run_gemm_reduce_scatter as runner
from benchmark_gemm_reduce_scatter import EventPair, TileLangRun as RsTileLangRun, _configure_hip, _mpi_abort
from gemm_reduce_scatter_conservative import _check_adapter_init
from allgather_gemm_scheduled import (PE_COUNT, THREADS, TILE_M, TILE_N, _check_shape,
                                      build_bound, schedule_parameters)


TARGET = "hip -mcpu=gfx936"
HOST_TO_DEVICE = runner.HOST_TO_DEVICE
DEVICE_TO_HOST = runner.DEVICE_TO_HOST


def _verify_jit_source(source: str, enabled: bool) -> None:
    """Fail closed if JIT lowering did not honor the requested pass config."""
    marker = "ipc_ag_half8_i"
    if not enabled:
        if marker in source:
            raise RuntimeError("--scalar requested but JIT HIP contains ipc_ag_half8_i")
        return
    lines = source.splitlines()
    pointer_lines = [line for line in lines if "tl::ipc::peer<std::byte>(ipc_view_a_shard" in line]
    copy_lines = [line for line in lines if "*(uint4*)" in line and "remote_source" in line]
    if marker not in source or len(pointer_lines) != 1 or len(copy_lines) != 1:
        raise RuntimeError("requested AG IPC half8 pass did not emit one hoisted uint4 remote copy")
    if lines.index(pointer_lines[0]) >= next(i for i, line in enumerate(lines) if marker in line):
        raise RuntimeError("AG remote base was not hoisted before the half8 payload loop")


def _dyadic_gathered(m: int, k: int, epoch: int) -> np.ndarray:
    shard_rows = m // PE_COUNT
    rows = np.arange(m, dtype=np.int32)
    source = rows // shard_rows
    q = np.arange(k, dtype=np.int32) % 4
    # an(source, globalrow, q, epoch) / 64; all inputs are exactly dyadic.
    values = source[:, None] + 1 + (rows[:, None] % 7) + epoch + q[None, :] - 2
    return (values.astype(np.float32) / 64).astype(np.float16)


def _dyadic_b(rank: int, n: int, k: int, epoch: int) -> np.ndarray:
    q = np.arange(k, dtype=np.int32) % 4
    global_col = rank * n + np.arange(n, dtype=np.int32)
    values = ((global_col[None, :] % 5) - 2 + epoch + ((q[:, None] * 3) % 5) - 2)
    return (values.astype(np.float32) / 64).astype(np.float16)


def _dyadic_output(rank: int, m: int, n: int, k: int, epoch: int) -> np.ndarray:
    """Direct analytic oracle; never uses a GPU result or an S-sized matmul."""
    shard_rows = m // PE_COUNT
    rows = np.arange(m, dtype=np.int32)
    source = rows // shard_rows
    global_col = rank * n + np.arange(n, dtype=np.int32)
    acc = np.zeros((m, n), dtype=np.int32)
    for q in range(4):
        an = source + 1 + (rows % 7) + epoch + q - 2
        bn = (global_col % 5) - 2 + epoch + ((q * 3) % 5) - 2
        acc += an[:, None] * bn[None, :]
    return (acc.astype(np.float32) * (k // 4) / 4096).astype(np.float16)


def _continuous_inputs(rank: int, m: int, n: int, k: int, epoch: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Small diagnostic path; regenerate every source deterministically."""
    if m > 2048:
        raise ValueError("--continuous is limited to S<=2048; dyadic is the full-shape oracle")
    shard_rows = m // PE_COUNT
    shards = []
    for source in range(PE_COUNT):
        shards.append(np.random.default_rng(1000 + 97 * epoch + source).uniform(-1, 1, (shard_rows, k)).astype(np.float16))
    b = np.random.default_rng(2000 + 97 * epoch + rank).uniform(-1, 1, (k, n)).astype(np.float16)
    gathered = np.concatenate(shards, axis=0)
    return shards[rank], b, gathered


def _inputs(rank: int, m: int, n: int, k: int, epoch: int, continuous: bool) -> tuple[np.ndarray, np.ndarray]:
    if continuous:
        shard, b, _ = _continuous_inputs(rank, m, n, k, epoch)
        return shard, b
    gathered = _dyadic_gathered(m, k, epoch)
    return gathered[rank * (m // PE_COUNT):(rank + 1) * (m // PE_COUNT)], _dyadic_b(rank, n, k, epoch)


def _expected(rank: int, m: int, n: int, k: int, epoch: int, continuous: bool) -> tuple[np.ndarray, np.ndarray]:
    if continuous:
        _, b, gathered = _continuous_inputs(rank, m, n, k, epoch)
        return gathered, (gathered.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)
    return _dyadic_gathered(m, k, epoch), _dyadic_output(rank, m, n, k, epoch)


def _check_array(actual: np.ndarray, expected: np.ndarray, label: str, continuous: bool) -> None:
    if continuous:
        if np.allclose(actual, expected, rtol=1e-3, atol=1e-3, equal_nan=False):
            return
        raise AssertionError(f"{label}: continuous mismatch max_abs="
                             f"{np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))}")
    mismatch = np.flatnonzero(actual.view(np.uint16).reshape(-1) != expected.view(np.uint16).reshape(-1))
    if mismatch.size:
        i = int(mismatch[0])
        raise AssertionError(f"{label}: dyadic bit mismatch count={mismatch.size}, first={i}, "
                             f"actual=0x{actual.view(np.uint16).reshape(-1)[i]:04x}, "
                             f"expected=0x{expected.view(np.uint16).reshape(-1)[i]:04x}")


class Run:
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
            raise RuntimeError(f"requires exactly {PE_COUNT} ranks, got {self.ctx.world_size}")
        self.hip = _configure_hip()
        self.schedule = schedule_parameters(args.m, args.n, args.k, total_blocks=args.total_blocks,
                                            comm_blocks=args.comm_blocks)
        self.flag_words = self.schedule["flag_words"]
        self.allocations = [
            self.ctx.alloc("ag-gemm-a-shard", (args.m // PE_COUNT) * args.k * 2, fine_grained=True),
            self.ctx.alloc("ag-gemm-a", args.m * args.k * 2, fine_grained=False),
            self.ctx.alloc("ag-gemm-b", args.k * args.n * 2, fine_grained=False),
            self.ctx.alloc("ag-gemm-ready", self.flag_words * 4, fine_grained=True),
            self.ctx.alloc("ag-gemm-out", args.m * args.n * 2, fine_grained=False),
        ]
        self.a_shard, self.a, self.b, self.ready, self.out = self.allocations
        self.launched = False
        self.reader_safe = True
        self.residency = None
        self.residencies = {}
        self._initialize_epoch(0)
        self.kernel = self._compile()
        self.args_tuple = self._prebind_args()

    def _initialize_epoch(self, epoch: int) -> None:
        shard, b = _inputs(self.rank, self.args.m, self.args.n, self.args.k, epoch, self.args.continuous)
        runner._copy(self.hip, self.a_shard.data_ptr(), shard.ctypes.data, shard.nbytes, HOST_TO_DEVICE)
        runner._copy(self.hip, self.b.data_ptr(), b.ctypes.data, b.nbytes, HOST_TO_DEVICE)
        runner._check(self.hip.hipDeviceSynchronize(), "epoch input upload")
        self.ctx.sync()

    def _compile(self):
        import tilelang
        tilelang.disable_cache()
        _, bound = build_bound(self.args.m, self.args.n, self.args.k, total_blocks=self.args.total_blocks,
                              comm_blocks=self.args.comm_blocks)
        kernel = tilelang.compile(bound, target=TARGET, execution_backend="ctypes",
                                 pass_configs={"tl.enable_dcu_ipc_vectorize": not self.args.scalar})
        _check_adapter_init(kernel)
        source = kernel.get_kernel_source()
        _verify_jit_source(source, enabled=not self.args.scalar)
        if self.args.export_hip:
            self.args.export_hip.mkdir(parents=True, exist_ok=True)
            (self.args.export_hip / f"tilelang-ag-m{self.args.m}-n{self.args.n}-k{self.args.k}-rank{self.rank}.hip").write_text(source, encoding="utf-8")
        RsTileLangRun._require_full_residency(self, kernel)
        self.residencies["tilelang"] = dict(self.residency)
        return kernel

    def _prebind_args(self):
        funcs = [f for f in self.kernel.adapter.ir_module.functions.values() if hasattr(f, "params") and hasattr(f, "buffer_map")]
        if len(funcs) != 1:
            raise RuntimeError(f"expected one adapter PrimFunc, got {len(funcs)}")
        prim = funcs[0]
        def name(p):
            buf = prim.buffer_map.get(p)
            return str(buf.name) if buf is not None else str(getattr(p, "name", getattr(p, "name_hint", "")))
        names = tuple(name(p) for p in prim.params)
        expected = ("A_shard", "A", "B", "ready", "out", "ipc_view_a_shard", "ipc_view_ready", "__tl_ipc_rank", "__tl_ipc_world_size")
        if names != expected:
            raise RuntimeError(f"unexpected TileLang AG C-ABI order: {names}")
        lib = self.kernel.adapter.lib
        lib.call.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p]
        lib.call.restype = ctypes.c_int
        return (ctypes.c_void_p(self.a_shard.data_ptr()), ctypes.c_void_p(self.a.data_ptr()),
                ctypes.c_void_p(self.b.data_ptr()), ctypes.c_void_p(self.ready.data_ptr()), ctypes.c_void_p(self.out.data_ptr()),
                ctypes.c_void_p(self.a_shard.device_view().data_ptr()), ctypes.c_void_p(self.ready.device_view().data_ptr()),
                ctypes.c_int32(self.rank), ctypes.c_int32(self.ctx.world_size), ctypes.c_void_p(0))

    def prepare(self, poison: bool = False) -> None:
        if poison:
            runner._check(self.hip.hipMemset(ctypes.c_void_p(self.a.data_ptr()), 0xA5, self.args.m * self.args.k * 2), "gathered A poison")
            runner._check(self.hip.hipMemset(ctypes.c_void_p(self.out.data_ptr()), 0xA5, self.args.m * self.args.n * 2), "out poison")
        runner._check(self.hip.hipMemset(ctypes.c_void_p(self.ready.data_ptr()), 0, self.flag_words * 4), "ready reset")
        runner._check(self.hip.hipDeviceSynchronize(), "prepare completion")
        self.ctx.sync()

    def launch(self) -> None:
        self.launched, self.reader_safe = True, False
        status = self.kernel.adapter.lib.call(*self.args_tuple)
        if status:
            raise RuntimeError(f"TileLang C-ABI launch returned {status}")
        if self.hip.hipGetLastError():
            raise RuntimeError("TileLang launch reported HIP error")

    def complete_local(self) -> None:
        runner._check(self.hip.hipDeviceSynchronize(), "post-kernel local synchronize")
        self.reader_safe = True

    def check(self, epoch: int) -> None:
        gathered = np.empty((self.args.m, self.args.k), dtype=np.float16)
        out = np.empty((self.args.m, self.args.n), dtype=np.float16)
        runner._copy(self.hip, gathered.ctypes.data, self.a.data_ptr(), gathered.nbytes, DEVICE_TO_HOST)
        runner._copy(self.hip, out.ctypes.data, self.out.data_ptr(), out.nbytes, DEVICE_TO_HOST)
        runner._check(self.hip.hipDeviceSynchronize(), "copy full gathered/output")
        expected_a, expected_out = _expected(self.rank, self.args.m, self.args.n, self.args.k, epoch, self.args.continuous)
        # AllGather is a copy operation even in the continuous diagnostic;
        # preserve its exact FP16 bit pattern rather than allowing the GEMM
        # tolerance to conceal a stale or misrouted chunk.
        _check_array(gathered, expected_a, "gathered A", False)
        _check_array(out, expected_out, "GEMM output", self.args.continuous)

    def finish_round(self) -> None:
        self.ctx.sync()

    def close(self) -> None:
        if self.launched and not self.reader_safe:
            os._exit(1)
        for allocation in reversed(self.allocations): allocation.close()
        self.ctx.close()
        runner._check(self.mpi.MPI_Finalize(), "MPI_Finalize")


def _one(state: Run, events: EventPair, timeout: float) -> float:
    state.prepare()
    elapsed = events.time_one(state.launch, timeout)
    state.complete_local()
    return elapsed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m", type=int, default=2048); p.add_argument("--n", type=int, default=512); p.add_argument("--k", type=int, default=2048)
    p.add_argument("--total-blocks", type=int, default=80); p.add_argument("--comm-blocks", type=int)
    p.add_argument("--scalar", action="store_true", help="disable the verified AG IPC half8 pass")
    p.add_argument("--continuous", action="store_true", help="small S<=2048 NumPy-matmul diagnostic")
    p.add_argument("--warmup", type=int, default=5); p.add_argument("--repeats", type=int, default=20); p.add_argument("--timeout-seconds", type=float, default=60)
    p.add_argument("--json-dir", type=Path, default=HERE / "benchmark_results"); p.add_argument("--export-hip", type=Path)
    args = p.parse_args()
    _check_shape(args.m, args.n, args.k)
    if args.comm_blocks is None: args.comm_blocks = schedule_parameters(args.m, args.n, args.k)["comm_blocks"]
    if args.warmup < 0 or args.repeats <= 0 or args.timeout_seconds <= 0: raise ValueError("invalid timing arguments")
    state = events = None
    try:
        state = Run(args); events = EventPair(state.hip)
        print(f"rank={state.rank} AG+GEMM grid={state.schedule} residency={state.residency}", flush=True)
        verification = []
        # Two poisoned epochs prove both the gathered workspace and output are
        # fully overwritten; checks run after local sync but before ctx.sync.
        for epoch in (0, 1):
            state._initialize_epoch(epoch); state.prepare(poison=True); state.launch(); state.complete_local(); state.check(epoch)
            verification.append({"epoch": epoch, "status": "passed"}); state.finish_round()
        state._initialize_epoch(0)
        for _ in range(args.warmup): _one(state, events, args.timeout_seconds); state.finish_round()
        samples = []
        for _ in range(args.repeats): samples.append(_one(state, events, args.timeout_seconds)); state.finish_round()
        state.prepare(poison=True); state.launch(); state.complete_local(); state.check(0); verification.append({"epoch": 0, "post_timing": "passed"}); state.finish_round()
    except BaseException as error:
        mpi = state.mpi if state is not None else ctypes.CDLL("libmpi.so")
        print(f"AG+GEMM abort: {error}", flush=True); _mpi_abort(mpi)
    finally:
        if events is not None: events.close()
        if state is not None: state.close()
    args.json_dir.mkdir(parents=True, exist_ok=True)
    output = args.json_dir / f"ag-gemm-tilelang-m{args.m}-n{args.n}-k{args.k}-rank{state.rank}.json"
    doc = {"schema": "ag-gemm-per-pe-hip-event-v1", "rank": state.rank,
           "shape": {"m": args.m, "n": args.n, "k": args.k, "pe_count": PE_COUNT},
           "grid": {"total_blocks": args.total_blocks, "comm_blocks": args.comm_blocks, "compute_blocks": state.schedule["compute_blocks"]},
           "allocations": "A_shard/ready FineGrained; gathered A/B/out DeviceDefault",
           "scalar": args.scalar, "continuous": args.continuous, "warmup": args.warmup, "repeats": args.repeats,
           "residency": state.residency, "verification": verification,
           "timing": {"samples_ms": samples, "median_ms": float(statistics.median(samples))},
           "notes": ["raw per-rank HIP event samples; aggregate ranks by elementwise max", "dyadic checks are full bitwise gathered-A and output checks"]}
    output.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"AG+GEMM benchmark rank={state.rank} json={output}", flush=True)


if __name__ == "__main__":
    main()
