"""Per-PE HIP-event benchmark for the GEMM + ReduceScatter experiments.

This is deliberately a benchmark *harness*, not a kernel implementation.  By
default ``--schedule single`` preserves the original one-CTA serial-tile
diagnostic.  ``--schedule matched`` selects the bounded 80-CTA TileLang
candidate (48 compute + 32 communication CTAs), whose host path fail-closes
unless the current GPU can fully reside its 512-thread/64KiB grid.  Neither
schedule is a general persistent-kernel claim.  Use the optional
``--implementation handwritten`` branch to measure an already-built
dcu_fused_gemm IPC shared library with the same input and timing definition.
``--diagnostic-hip PATH`` is a separate, explicitly labelled opt-in branch for
an exported matched HIP source artifact; it never represents that artifact as
TileLang-generated code.

Run one process on each of four DCUs, for example::

  mpirun -n 4 --bind-to none python benchmark_gemm_reduce_scatter.py \\
      --implementation tilelang --schedule matched --m 2048 --n 2048 --k 512

Every rank writes an independent JSON file.  To compare ranks, take the
elementwise maximum of their ``samples_ms`` first, then compute median/p10/p90;
do not average the four ranks' summaries.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

# Keep imports relative to this file so it can be launched from an allocation's
# arbitrary working directory.
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_gemm_reduce_scatter as runner
from gemm_reduce_scatter import PE_COUNT, THREADS, TILE_M, TILE_N, _check_shape
from gemm_rs_diagnostic_hip_abi import diagnostic_hip_arguments
from gemm_reduce_scatter_conservative import _check_adapter_init, _checked_launch


TARGET = "hip -mcpu=gfx936"
READY_NOT_READY = 600  # hipErrorNotReady
DIAGNOSTIC_HIP_ADAPTER = HERE / "gemm_rs_diagnostic_hip_adapter.cpp"


def _configure_hip():
    """Return the established ctypes HIP handle, extended for elapsed timing."""
    hip = runner._hip()
    hip.hipEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
    hip.hipEventElapsedTime.restype = ctypes.c_int
    hip.hipGetLastError.argtypes = []
    hip.hipGetLastError.restype = ctypes.c_int
    hip.hipGetErrorString.argtypes = [ctypes.c_int]
    hip.hipGetErrorString.restype = ctypes.c_char_p
    return hip


def _check_last_error(hip, what: str) -> None:
    code = hip.hipGetLastError()
    if code:
        message = hip.hipGetErrorString(code)
        detail = message.decode("utf-8", errors="replace") if message else "unknown HIP error"
        raise RuntimeError(f"{what}: HIP status {code}: {detail}")


class EventPair:
    """Two default-stream events, used for exactly one kernel launch at a time."""

    def __init__(self, hip):
        self.hip = hip
        self.start = ctypes.c_void_p()
        self.end = ctypes.c_void_p()
        runner._check(hip.hipEventCreate(ctypes.byref(self.start)), "hipEventCreate(start)")
        try:
            runner._check(hip.hipEventCreate(ctypes.byref(self.end)), "hipEventCreate(end)")
        except BaseException:
            runner._check(hip.hipEventDestroy(self.start), "hipEventDestroy(start)")
            raise

    def close(self) -> None:
        runner._check(self.hip.hipEventDestroy(self.end), "hipEventDestroy(end)")
        runner._check(self.hip.hipEventDestroy(self.start), "hipEventDestroy(start)")

    def time_one(self, launch, timeout_seconds: float) -> float:
        runner._check(self.hip.hipEventRecord(self.start, None), "hipEventRecord(start)")
        launch()
        runner._check(self.hip.hipEventRecord(self.end, None), "hipEventRecord(end)")
        deadline = time.monotonic() + timeout_seconds
        while True:
            status = self.hip.hipEventQuery(self.end)
            if status != READY_NOT_READY:
                runner._check(status, "hipEventQuery(end)")
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"GEMM + ReduceScatter kernel exceeded {timeout_seconds:g}s")
            time.sleep(0.005)
        elapsed = ctypes.c_float()
        runner._check(
            self.hip.hipEventElapsedTime(ctypes.byref(elapsed), self.start, self.end),
            "hipEventElapsedTime",
        )
        return float(elapsed.value)


def _rank_from_environment() -> int:
    for name in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "MV2_COMM_WORLD_RANK"):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    raise RuntimeError("could not determine MPI rank from the launcher environment")


def _percentile(values: list[float], q: float) -> float:
    # ``method`` was added after some old-tree Python environments; this is
    # sufficient for an explicit, JSON-friendly summary and does not affect the
    # raw per-iteration samples used for cross-rank aggregation.
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _dump_partial_failure(directory: Path, actual: np.ndarray, expected: np.ndarray, *, rank: int,
                          m: int, n: int, k: int, mode: str) -> None:
    """Persist rank-local evidence without changing the numerical verdict."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        actual_path = directory / f"partial-actual-rank{rank}.npy"
        expected_path = directory / f"partial-expected-rank{rank}.npy"
        np.save(actual_path, actual)
        np.save(expected_path, expected)
        # Dyadic verification is bitwise.  Continuous mode keeps the runner's
        # existing 1e-3 allclose tolerance, while also retaining exact-bit
        # count for postmortem comparison.
        bit_mismatch = actual.view(np.uint16) != expected.view(np.uint16)
        mismatch = bit_mismatch if mode == "dyadic" else ~np.isclose(
            actual, expected, rtol=1e-3, atol=1e-3, equal_nan=False,
        )
        diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
        document = {
            "schema": "gemm-rs-partial-failure-v1",
            "rank": rank,
            "shape": {"m": m, "n": n, "k": k, "tile_m": TILE_M, "tile_n": TILE_N},
            "mode": mode,
            "actual": actual_path.name,
            "expected": expected_path.name,
            "mismatch_count": int(np.count_nonzero(mismatch)),
            "bit_mismatch_count": int(np.count_nonzero(bit_mismatch)),
            "nan_count": int(np.count_nonzero(np.isnan(actual))),
            "max_abs": float(np.nanmax(diff, initial=0.0)),
            # These retain the tile-major layout used by the partial workspace
            # and expose whether a failure clusters by tile, row, or column.
            "mismatch_by_tile": np.count_nonzero(mismatch, axis=(1, 2)).tolist(),
            "mismatch_by_tile_row": np.count_nonzero(mismatch, axis=2).tolist(),
            "mismatch_by_tile_col": np.count_nonzero(mismatch, axis=1).tolist(),
        }
        (directory / f"partial-error-rank{rank}.json").write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        print(f"rank={rank}: saved partial failure evidence in {directory}", flush=True)
    except BaseException as error:
        # The original numerical error must still drive the group abort if a
        # diagnostic filesystem problem occurs.
        print(f"rank={rank}: could not save partial failure evidence: {error}", flush=True)


def _dump_reduce_scatter_failure(directory: Path, actual: np.ndarray, expected: np.ndarray,
                                 partial_actual: np.ndarray, partial_expected: np.ndarray, *,
                                 rank: int, m: int, n: int, k: int, mode: str) -> None:
    """Persist output plus the already-copied partials without changing the verdict."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        actual_path = directory / f"out-actual-rank{rank}.npy"
        expected_path = directory / f"out-expected-rank{rank}.npy"
        partial_actual_path = directory / f"partial-actual-rank{rank}.npy"
        partial_expected_path = directory / f"partial-expected-rank{rank}.npy"
        np.save(actual_path, actual)
        np.save(expected_path, expected)
        # Reuse the host arrays copied for the preceding partial check.  This
        # makes a strict RS failure directly comparable with hand-written runs
        # without an extra device read or altered synchronization.
        np.save(partial_actual_path, partial_actual)
        np.save(partial_expected_path, partial_expected)
        bit_mismatch = actual.view(np.uint16) != expected.view(np.uint16)
        mismatch = bit_mismatch if mode == "dyadic" else ~np.isclose(
            actual, expected, rtol=1e-3, atol=1e-3, equal_nan=False,
        )
        diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
        document = {
            "schema": "gemm-rs-result-failure-v1",
            "label": "reduce-scatter",
            "rank": rank,
            "shape": {"m": m, "n": n, "k": k, "tile_m": TILE_M, "tile_n": TILE_N},
            "mode": mode,
            "actual": actual_path.name,
            "expected": expected_path.name,
            "partial_actual": partial_actual_path.name,
            "partial_expected": partial_expected_path.name,
            "mismatch_count": int(np.count_nonzero(mismatch)),
            "bit_mismatch_count": int(np.count_nonzero(bit_mismatch)),
            "nan_count": int(np.count_nonzero(np.isnan(actual))),
            "max_abs": float(np.nanmax(diff, initial=0.0)),
            "mismatch_by_row": np.count_nonzero(mismatch, axis=1).tolist(),
            "mismatch_by_col": np.count_nonzero(mismatch, axis=0).tolist(),
            "partial_bit_mismatch_count": int(np.count_nonzero(
                partial_actual.view(np.uint16) != partial_expected.view(np.uint16)
            )),
        }
        (directory / f"reduce-scatter-error-rank{rank}.json").write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        print(f"rank={rank}: saved reduce-scatter failure evidence in {directory}", flush=True)
    except BaseException as error:
        # Preserve the original numerical assertion and MPI abort semantics.
        print(f"rank={rank}: could not save reduce-scatter failure evidence: {error}", flush=True)


def _full_check(hip, partial_ptr: int, out_ptr: int, rank: int, m: int, n: int, k: int,
                mode: str, failure_dump_dir: Path | None = None) -> dict:
    """The original full CPU oracle, deliberately opt-in for large shapes."""
    tile_count = (m // TILE_M) * (n // TILE_N)
    partial_actual = np.empty((tile_count, TILE_M, TILE_N), dtype=np.float16)
    out_actual = np.empty((m // PE_COUNT, n), dtype=np.float16)
    runner._copy(hip, partial_actual.ctypes.data, partial_ptr, partial_actual.nbytes, runner.DEVICE_TO_HOST)
    runner._copy(hip, out_actual.ctypes.data, out_ptr, out_actual.nbytes, runner.DEVICE_TO_HOST)
    runner._check(hip.hipDeviceSynchronize(), "full result copies")
    local_partial = runner._partial_reference(rank, m, n, k, mode)
    partial_expected = runner._tile_view(local_partial)
    try:
        runner._check_result(partial_actual, partial_expected, "partial", mode, rank)
    except AssertionError:
        if failure_dump_dir is not None:
            _dump_partial_failure(
                failure_dump_dir, partial_actual, partial_expected,
                rank=rank, m=m, n=n, k=k, mode=mode,
            )
        raise
    reduced = sum(runner._partial_reference(peer, m, n, k, mode).astype(np.float32)
                  for peer in range(PE_COUNT))
    expected = reduced.astype(np.float16)[rank * (m // PE_COUNT):(rank + 1) * (m // PE_COUNT), :]
    try:
        runner._check_result(out_actual, expected, "reduce-scatter", mode, rank)
    except AssertionError:
        if failure_dump_dir is not None:
            _dump_reduce_scatter_failure(
                failure_dump_dir, out_actual, expected, partial_actual, partial_expected,
                rank=rank, m=m, n=n, k=k, mode=mode,
            )
        raise
    return {"mode": f"full-{mode}", "status": "passed"}


class TileLangRun:
    """Allocation, checked launch, and reader-safe reset for the pipeline prototype."""

    def __init__(self, args):
        from tilelang.distributed.ipc_runtime import IPCContext

        self.args = args
        self.mpi = ctypes.CDLL("libmpi.so")
        provided = ctypes.c_int()
        runner._check(self.mpi.MPI_Init_thread(None, None, 1, ctypes.byref(provided)), "MPI_Init_thread")
        self.ctx = IPCContext()
        self.rank = self.ctx.rank
        if self.ctx.world_size != PE_COUNT:
            raise RuntimeError(f"this prototype requires {PE_COUNT} MPI ranks, got {self.ctx.world_size}")
        self.hip = _configure_hip()
        self.tile_count = (args.m // TILE_M) * (args.n // TILE_N)
        self.matched = args.schedule == "matched"
        if self.matched:
            from gemm_reduce_scatter_scheduled import flag_layout, schedule_parameters

            self.schedule_parameters = schedule_parameters(
                args.m, args.n, args.total_blocks, args.comm_blocks,
            )
            self.flag_words = flag_layout(self.tile_count, args.comm_blocks)["flag_words"]
            # Match the hand-written allocation contract: local GEMM inputs
            # and output are DeviceDefault; only IPC payload/flags are shared
            # FineGrained allocations.
            self.allocations = [
                self.ctx.alloc("gemm-rs-matched-a", args.m * args.k * 2, fine_grained=False),
                self.ctx.alloc("gemm-rs-matched-b", args.k * args.n * 2, fine_grained=False),
                self.ctx.alloc("gemm-rs-matched-partial", self.tile_count * TILE_M * TILE_N * 2),
                self.ctx.alloc("gemm-rs-matched-ready", self.flag_words * 4),
                self.ctx.alloc("gemm-rs-matched-out", (args.m // PE_COUNT) * args.n * 2, fine_grained=False),
            ]
        else:
            self.schedule_parameters = None
            self.flag_words = self.tile_count * PE_COUNT
            # Preserve the historical single-CTA allocation path verbatim.
            self.allocations = [
                self.ctx.alloc("gemm-rs-bench-a", args.m * args.k * 2),
                self.ctx.alloc("gemm-rs-bench-b", args.k * args.n * 2),
                self.ctx.alloc("gemm-rs-bench-partial", self.tile_count * TILE_M * TILE_N * 2),
                self.ctx.alloc("gemm-rs-bench-ready", self.flag_words * 4),
                self.ctx.alloc("gemm-rs-bench-out", (args.m // PE_COUNT) * args.n * 2),
            ]
        self.a, self.b, self.partial, self.ready, self.out = self.allocations
        self.launched = False
        self.reader_safe = False
        self.residency = None
        self._initialize_inputs()
        self.kernel = self._compile()

    def _initialize_inputs(self) -> None:
        a_host, b_host = runner._rank_inputs(
            self.rank, self.args.m, self.args.n, self.args.k, self.args.input_mode,
        )
        runner._copy(self.hip, self.a.data_ptr(), a_host.ctypes.data, a_host.nbytes, runner.HOST_TO_DEVICE)
        runner._copy(self.hip, self.b.data_ptr(), b_host.ctypes.data, b_host.nbytes, runner.HOST_TO_DEVICE)
        # partial/out are fully overwritten by a completed launch; ready is
        # reset before every launch below.  Initialization happens before any
        # timing and provides a clean first iteration.
        runner._check(self.hip.hipMemset(ctypes.c_void_p(self.ready.data_ptr()), 0,
                                         self.flag_words * 4), "initial ready reset")
        runner._check(self.hip.hipDeviceSynchronize(), "TileLang input initialization")
        self.ctx.sync()

    def _compile(self):
        import tilelang

        if self.matched:
            from gemm_reduce_scatter_scheduled import build_bound
            _, bound = build_bound(
                self.args.m, self.args.n, self.args.k,
                total_blocks=self.args.total_blocks, comm_blocks=self.args.comm_blocks,
            )
        else:
            from gemm_reduce_scatter import build_bound
            _, bound = build_bound(self.args.m, self.args.n, self.args.k)
        # Do not reuse a host-side compiled-kernel cache across runs.  Compile
        # once for this rank, outside both warmup and measured samples.
        tilelang.disable_cache()
        kernel = tilelang.compile(bound, target=TARGET, execution_backend="ctypes")
        _check_adapter_init(kernel)
        if self.args.export_hip is not None:
            self.args.export_hip.mkdir(parents=True, exist_ok=True)
            kind = "matched" if self.matched else "pipeline"
            path = self.args.export_hip / f"tilelang-{kind}-m{self.args.m}-n{self.args.n}-k{self.args.k}-rank{self.rank}.hip"
            path.write_text(kernel.get_kernel_source(), encoding="utf-8")
        if self.matched:
            self._require_full_residency(kernel)
        return kernel

    def _require_full_residency(self, kernel) -> None:
        """Fail closed unless the actual device can co-reside the whole grid."""
        source = kernel.get_kernel_source()
        # TileLang's ctypes adapter exposes the generated device symbol through
        # its CDLL.  Reject an unrecognizable source rather than silently
        # assuming some wrapper pointer has the desired occupancy.
        import re
        names = re.findall(r'extern "C" __global__ void ([A-Za-z_][A-Za-z0-9_]*)\s*\(', source)
        if len(names) != 1:
            raise RuntimeError(f"matched occupancy check requires one exported kernel symbol, found {names}")
        try:
            function = ctypes.cast(getattr(kernel.adapter.lib, names[0]), ctypes.c_void_p)
        except AttributeError as error:
            raise RuntimeError(f"matched occupancy check could not resolve {names[0]} in the ctypes adapter") from error
        if not function.value:
            raise RuntimeError("matched occupancy check resolved a null kernel function pointer")
        self._require_full_residency_function(function)

    def _require_full_residency_function(self, function) -> None:
        """Apply the matched liveness check to a resolved HIP kernel symbol."""
        self.hip.hipOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t,
        ]
        self.hip.hipOccupancyMaxActiveBlocksPerMultiprocessor.restype = ctypes.c_int
        active = ctypes.c_int()
        runner._check(
            self.hip.hipOccupancyMaxActiveBlocksPerMultiprocessor(
                ctypes.byref(active), function, THREADS, 64 * 1024,
            ), "hipOccupancyMaxActiveBlocksPerMultiprocessor",
        )
        if active.value < 1:
            raise RuntimeError("matched 512-thread/64KiB kernel cannot reside on any CU")
        helper_source = HERE / "gemm_rs_occupancy_probe.cpp"
        helper_lib = Path(tempfile.gettempdir()) / f"gemm-rs-occupancy-{os.getpid()}.so"
        try:
            subprocess.run(
                ["hipcc", "-shared", "-fPIC", str(helper_source), "-o", str(helper_lib)],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            probe = ctypes.CDLL(str(helper_lib))
            probe.gemm_rs_scheduled_cu_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
            probe.gemm_rs_scheduled_cu_count.restype = ctypes.c_int
            cu_count = ctypes.c_int()
            runner._check(probe.gemm_rs_scheduled_cu_count(ctypes.byref(cu_count)), "hipGetDeviceProperties")
        finally:
            if helper_lib.exists():
                helper_lib.unlink()
        capacity = active.value * cu_count.value
        if cu_count.value <= 0 or capacity < self.args.total_blocks:
            raise RuntimeError(
                "matched grid is not fully resident: "
                f"CU={cu_count.value}, active_per_CU={active.value}, capacity={capacity}, "
                f"required={self.args.total_blocks}",
            )
        self.residency = {
            "cu_count": cu_count.value,
            "active_blocks_per_cu": active.value,
            "resident_capacity": capacity,
            "required_blocks": self.args.total_blocks,
            "threads_per_block": THREADS,
            "dynamic_shared_bytes": 64 * 1024,
        }

    def prepare(self) -> None:
        # This follows a completed event with device synchronization and a
        # cross-PE context sync.  Therefore no remote reader from the previous
        # launch can observe a reused ready flag.  Both reset and sync are
        # deliberately outside the HIP event interval.
        runner._check(self.hip.hipMemset(ctypes.c_void_p(self.ready.data_ptr()), 0,
                                         self.flag_words * 4), "ready reset")
        runner._check(self.hip.hipDeviceSynchronize(), "ready reset completion")
        self.ctx.sync()

    def launch(self) -> None:
        self.launched = True
        self.reader_safe = False
        _checked_launch(
            self.kernel, self.hip, self.a, self.b, self.partial, self.ready, self.out,
            self.partial.device_view(), self.ready.device_view(), self.rank, self.ctx.world_size,
        )

    def complete(self) -> None:
        runner._check(self.hip.hipDeviceSynchronize(), "post-kernel device synchronize")
        self.ctx.sync()
        self.reader_safe = True

    @property
    def partial_ptr(self) -> int:
        return self.partial.data_ptr()

    @property
    def out_ptr(self) -> int:
        return self.out.data_ptr()

    def close(self) -> None:
        if self.launched and not self.reader_safe:
            print(f"rank={self.rank}: kernel not reader-safe; bypassing collective free", flush=True)
            os._exit(1)
        for allocation in reversed(self.allocations):
            allocation.close()
        self.ctx.close()
        runner._check(self.mpi.MPI_Finalize(), "MPI_Finalize")

    def abort(self, reason: str) -> None:
        """Terminate all ranks before any rank can enter a mismatched teardown."""
        print(f"rank={self.rank}: aborting benchmark group: {reason}", flush=True)
        _mpi_abort(self.mpi)


class DiagnosticHipRun(TileLangRun):
    """TileLang IPC allocations driven by an explicitly external HIP artifact.

    This branch deliberately does not call TileLang compilation.  It is only
    constructed by ``--diagnostic-hip`` and therefore cannot silently replace
    the normal TileLang path.
    """

    def _compile(self):
        source_path = self.args.diagnostic_hip
        assert source_path is not None
        source = source_path.read_text(encoding="utf-8")
        match = re.findall(
            r'extern\s+"C"\s+__global__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^()]*)\)',
            source,
        )
        if len(match) != 1:
            raise RuntimeError(
                "diagnostic HIP source must export exactly one extern-C __global__ kernel; "
                f"found {[name for name, _ in match]}"
            )
        self.diagnostic_symbol, parameters = match[0]
        abi_parameters = [part.strip() for part in parameters.split(",") if part.strip()]
        if len(abi_parameters) != 9:
            raise RuntimeError(
                "diagnostic HIP source does not have the supported current nine-argument IPC ABI"
            )
        expected_types = ("half_t", "half_t", "uint64_t", "uint64_t", "half_t",
                          "half_t", "uint", "int", "int")
        if any(not re.search(rf"\b{re.escape(expected)}\b", actual)
               for actual, expected in zip(abi_parameters, expected_types)):
            raise RuntimeError(
                "diagnostic HIP source ABI differs from the supported "
                "(half_t*, half_t*, uint64_t*, uint64_t*, half_t*, half_t*, uint*, int, int) signature"
            )
        self.diagnostic_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
        self.diagnostic_lib_path = self._compile_diagnostic_adapter(source_path, self.diagnostic_symbol)
        self.diagnostic_lib = ctypes.CDLL(str(self.diagnostic_lib_path), mode=ctypes.RTLD_GLOBAL)
        self.diagnostic_lib.gemm_rs_diagnostic_hip_launch.argtypes = (
            [ctypes.c_void_p] * 7 + [ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p]
        )
        self.diagnostic_lib.gemm_rs_diagnostic_hip_launch.restype = ctypes.c_int
        try:
            function = ctypes.cast(getattr(self.diagnostic_lib, self.diagnostic_symbol), ctypes.c_void_p)
        except AttributeError as error:
            raise RuntimeError(
                f"diagnostic HIP adapter did not export kernel symbol {self.diagnostic_symbol!r}"
            ) from error
        if not function.value:
            raise RuntimeError("diagnostic HIP occupancy check resolved a null kernel function pointer")
        self._require_full_residency_function(function)
        return None

    def _compile_diagnostic_adapter(self, source_path: Path, symbol: str) -> Path:
        """Build only the opt-in C ABI wrapper plus the supplied source artifact."""
        from tilelang.contrib.rocm import find_rocm_path, get_rocm_arch
        from tilelang.env import COMPOSABLE_KERNEL_INCLUDE_DIR, TILELANG_TEMPLATE_PATH

        if not DIAGNOSTIC_HIP_ADAPTER.is_file():
            raise FileNotFoundError(DIAGNOSTIC_HIP_ADAPTER)
        build_dir = Path(tempfile.mkdtemp(prefix="gemm-rs-diagnostic-hip-"))
        self._diagnostic_build_dir = build_dir
        library = build_dir / "gemm-rs-diagnostic-hip.so"
        command = [
            "hipcc", "-std=c++17", "-O3", "-fPIC", "--gpu-max-threads-per-block=256",
            f"--offload-arch={get_rocm_arch(find_rocm_path())}", "--shared",
            "--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11",
            str(DIAGNOSTIC_HIP_ADAPTER),
            f'-DDIAGNOSTIC_HIP_SOURCE="{source_path}"',
            f"-DDIAGNOSTIC_HIP_KERNEL={symbol}",
            f"-DDIAGNOSTIC_HIP_GRID_X={self.args.total_blocks}",
            "-I" + COMPOSABLE_KERNEL_INCLUDE_DIR,
            "-I" + os.path.join(TILELANG_TEMPLATE_PATH, "runtime/dcu_ipc/include"),
            "-I" + TILELANG_TEMPLATE_PATH,
            "-o", str(library),
        ]
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if completed.returncode:
            raise RuntimeError(
                "diagnostic HIP adapter compilation failed for "
                f"{source_path}:\n{completed.stderr}"
            )
        return library

    def launch(self) -> None:
        self.launched = True
        self.reader_safe = False
        launch_args = diagnostic_hip_arguments(
            self.a, self.b, self.partial, self.ready, self.out, self.rank, self.ctx.world_size,
        )
        status = self.diagnostic_lib.gemm_rs_diagnostic_hip_launch(
            *[ctypes.c_void_p(value) for value in launch_args[:7]],
            ctypes.c_int32(launch_args[7]), ctypes.c_int32(launch_args[8]), ctypes.c_void_p(0),
        )
        if status:
            message = self.hip.hipGetErrorString(status)
            detail = message.decode("utf-8", errors="replace") if message else "unknown HIP error"
            raise RuntimeError(f"diagnostic HIP launch failed ({status}): {detail}")

    def close(self) -> None:
        try:
            super().close()
        finally:
            build_dir = getattr(self, "_diagnostic_build_dir", None)
            if build_dir is not None:
                shutil.rmtree(build_dir, ignore_errors=True)


class HandwrittenRun:
    """C-ABI adapter for a prebuilt dcu_fused_gemm ReduceScatter library."""

    def __init__(self, args):
        if args.reference_library is None or not args.reference_library.is_absolute():
            raise ValueError("--reference-library must be an absolute path for --implementation handwritten")
        expected_name = f"libfused_gemm_rs_m{args.m}_n{args.n}_k{args.k}_ipc.so"
        if args.reference_library.name != expected_name:
            raise ValueError(
                "handwritten reference-library shape mismatch: "
                f"CLI requires {expected_name}, got {args.reference_library.name}; "
                "refusing ipc_init/hipMemcpy because this ABI has fixed allocations"
            )
        if not args.reference_library.is_file():
            raise FileNotFoundError(args.reference_library)
        self.args = args
        self.rank = _rank_from_environment()
        self.hip = _configure_hip()
        self.lib = ctypes.CDLL(str(args.reference_library), mode=ctypes.RTLD_GLOBAL)
        for name in ("ipc_init", "ipc_barrier", "ipc_close"):
            fn = getattr(self.lib, name)
            fn.argtypes = []
            fn.restype = None
        for name in ("ipc_a_pointer", "ipc_b_pointer", "ipc_output_pointer", "ipc_work_pointer"):
            fn = getattr(self.lib, name)
            fn.argtypes = []
            fn.restype = ctypes.c_void_p
        self.lib.ipc_prepare.argtypes = [ctypes.c_int]
        self.lib.ipc_prepare.restype = None
        self.lib.ipc_launch.argtypes = [ctypes.c_int, ctypes.c_void_p]
        self.lib.ipc_launch.restype = None
        # This library owns MPI and IPC allocations; do not construct a
        # TileLang IPCContext in this branch.
        self.lib.ipc_init()
        self._initialize_inputs()
        self.launched = False
        self.reader_safe = True

    def _initialize_inputs(self) -> None:
        a_host, b_host = runner._rank_inputs(
            self.rank, self.args.m, self.args.n, self.args.k, self.args.input_mode,
        )
        a_ptr = self.lib.ipc_a_pointer()
        b_ptr = self.lib.ipc_b_pointer()
        if not a_ptr or not b_ptr:
            raise RuntimeError("handwritten IPC library returned a null input pointer")
        runner._copy(self.hip, a_ptr, a_host.ctypes.data, a_host.nbytes, runner.HOST_TO_DEVICE)
        runner._copy(self.hip, b_ptr, b_host.ctypes.data, b_host.nbytes, runner.HOST_TO_DEVICE)
        runner._check(self.hip.hipDeviceSynchronize(), "handwritten input initialization")
        self.lib.ipc_barrier()

    def prepare(self) -> None:
        # ipc_prepare includes its own cross-PE boundaries and flag reset.  It
        # is intentionally called before recording the start event.
        self.lib.ipc_prepare(0)

    def launch(self) -> None:
        self.launched = True
        self.reader_safe = False
        self.lib.ipc_launch(self.args.comm_blocks, None)
        _check_last_error(self.hip, "handwritten ipc_launch")

    def complete(self) -> None:
        runner._check(self.hip.hipDeviceSynchronize(), "handwritten post-kernel device synchronize")
        # The reference kernel has its own completion handshake, but this host
        # barrier makes the next prepare/reset a complete-round boundary too.
        self.lib.ipc_barrier()
        self.reader_safe = True

    @property
    def partial_ptr(self) -> int:
        value = self.lib.ipc_work_pointer()
        if not value:
            raise RuntimeError("handwritten IPC library returned a null work pointer")
        return value

    @property
    def out_ptr(self) -> int:
        value = self.lib.ipc_output_pointer()
        if not value:
            raise RuntimeError("handwritten IPC library returned a null output pointer")
        return value

    def close(self) -> None:
        if self.launched and not self.reader_safe:
            print(f"rank={self.rank}: handwritten kernel not reader-safe; bypassing ipc_close", flush=True)
            os._exit(1)
        self.lib.ipc_close()

    def abort(self, reason: str) -> None:
        """The reference owns MPI, but its process group still needs one abort."""
        print(f"rank={self.rank}: aborting benchmark group: {reason}", flush=True)
        _mpi_abort(ctypes.CDLL("libmpi.so"))


def _mpi_abort(mpi) -> None:
    """Best-effort MPI_Abort followed by _exit if the MPI implementation returns.

    MPI_Comm_f2c(0) is MPI_COMM_WORLD in the Fortran ABI used by the deployed
    HPC-X/OpenMPI stack.  The fallback _exit still makes mpirun tear down the
    job if a nonstandard MPI rejects the conversion.
    """
    try:
        mpi.MPI_Comm_f2c.argtypes = [ctypes.c_int]
        mpi.MPI_Comm_f2c.restype = ctypes.c_void_p
        mpi.MPI_Abort.argtypes = [ctypes.c_void_p, ctypes.c_int]
        mpi.MPI_Abort.restype = ctypes.c_int
        mpi.MPI_Abort(mpi.MPI_Comm_f2c(0), 1)
    finally:
        os._exit(1)


def _one_iteration(state, events: EventPair, timeout_seconds: float, wall_clock: bool) -> tuple[float, float | None]:
    wall_start = time.perf_counter() if wall_clock else None
    state.prepare()
    event_ms = events.time_one(state.launch, timeout_seconds)
    state.complete()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0 if wall_start is not None else None
    return event_ms, wall_ms


def run(args) -> Path:
    if args.schedule == "matched":
        from gemm_reduce_scatter_scheduled import _check_shape as check_matched_shape
        check_matched_shape(args.m, args.n, args.k)
    else:
        _check_shape(args.m, args.n, args.k)
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("--warmup must be nonnegative and --repeats must be positive")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    if args.diagnostic_hip is not None:
        if args.implementation != "tilelang":
            raise ValueError("--diagnostic-hip is a separate external-artifact branch; use --implementation tilelang")
        if args.schedule != "matched":
            raise ValueError("--diagnostic-hip currently supports only the exported matched 80-CTA ABI")
        if args.total_blocks != 80 or args.comm_blocks != 32:
            raise ValueError(
                "--diagnostic-hip currently requires the exported matched 80-CTA split "
                "(--total-blocks 80 --comm-blocks 32)"
            )
        args.diagnostic_hip = args.diagnostic_hip.expanduser().resolve()
        if not args.diagnostic_hip.is_file() or args.diagnostic_hip.suffix != ".hip":
            raise ValueError("--diagnostic-hip must name an existing exported .hip file")
    if args.failure_dump_dir is not None:
        dump_root = (REPO_ROOT / "build-old").resolve()
        requested = args.failure_dump_dir.resolve()
        if not requested.is_relative_to(dump_root):
            raise ValueError(f"--failure-dump-dir must be under {dump_root}")
        args.failure_dump_dir = requested
    if args.implementation == "handwritten" and not (0 < args.comm_blocks < 80):
        raise ValueError("--comm-blocks must be in [1, 79] for the handwritten reference")
    if args.implementation == "handwritten" and args.schedule != "single":
        raise ValueError("--schedule matched is a TileLang candidate, not the handwritten C-ABI branch")
    if args.diagnostic_hip is not None:
        state = DiagnosticHipRun(args)
        implementation_label = "diagnostic-hip"
    elif args.implementation == "tilelang":
        state = TileLangRun(args)
        implementation_label = "tilelang"
    else:
        state = HandwrittenRun(args)
        implementation_label = "handwritten"
    events = EventPair(state.hip)
    try:
        # Correctness launch, compilation, allocation and input copies do not
        # contribute to samples.  A failed precheck aborts before warmup and
        # leaves no performance JSON behind.
        _, _ = _one_iteration(state, events, args.timeout_seconds, False)
        precheck = _full_check(state.hip, state.partial_ptr, state.out_ptr, state.rank,
                               args.m, args.n, args.k, args.input_mode, args.failure_dump_dir)

        for _ in range(args.warmup):
            _one_iteration(state, events, args.timeout_seconds, False)
        samples_ms: list[float] = []
        wall_samples_ms: list[float] | None = [] if args.wall_clock else None
        for _ in range(args.repeats):
            event_ms, wall_ms = _one_iteration(state, events, args.timeout_seconds, args.wall_clock)
            samples_ms.append(event_ms)
            if wall_samples_ms is not None:
                assert wall_ms is not None
                wall_samples_ms.append(wall_ms)
        # Check the final measured launch too.  A performance document thus
        # represents a run whose input/output semantics held both before and
        # after the recorded samples.
        postcheck = _full_check(state.hip, state.partial_ptr, state.out_ptr, state.rank,
                                args.m, args.n, args.k, args.input_mode, args.failure_dump_dir)
    except BaseException as error:
        # A rank that fails numerical validation must not proceed to warmup,
        # timing, or collective allocation close while peers take another path.
        # This also guarantees that a failing run never writes a performance
        # document.
        state.abort(str(error))
        raise AssertionError("unreachable after MPI_Abort") from error
    finally:
        # An event timeout leaves reader_safe false, which intentionally exits
        # rather than collectively freeing IPC memory that a remote PE may read.
        events.close()
        state.close()

    args.json_dir.mkdir(parents=True, exist_ok=True)
    stem = (f"{args.json_prefix}-{implementation_label}-{args.schedule}-"
            f"m{args.m}-n{args.n}-k{args.k}-rank{state.rank}.json")
    output = args.json_dir / stem
    matched_tilelang = implementation_label == "tilelang" and args.schedule == "matched"
    diagnostic_hip = implementation_label == "diagnostic-hip"
    if args.implementation == "handwritten":
        grid = {"managed_by": "handwritten reference library",
                "requested_comm_blocks": args.comm_blocks}
        flags = {"managed_by": "handwritten reference library"}
        allocation_note = "handwritten reference library owns its established allocations"
        residency = None
    elif matched_tilelang or diagnostic_hip:
        grid = {"total_blocks": args.total_blocks, "comm_blocks": args.comm_blocks,
                "compute_blocks": args.total_blocks - args.comm_blocks}
        flags = {"words": state.flag_words, "tile_ready_words": state.tile_count * PE_COUNT,
                 "peer_done_words": PE_COUNT, "block_done_words": args.comm_blocks}
        allocation_note = (
            "diagnostic external HIP: A/B/out DeviceDefault; partial/ready FineGrained"
            if diagnostic_hip else "matched: A/B/out DeviceDefault; partial/ready FineGrained"
        )
        residency = state.residency
    else:
        grid = {"total_blocks": 1, "comm_blocks": 0, "compute_blocks": 1}
        flags = {"words": state.flag_words, "tile_ready_words": state.flag_words}
        allocation_note = "single: historical FineGrained allocations"
        residency = state.residency
    document = {
        "schema": "gemm-rs-per-pe-hip-event-v1",
        "implementation": implementation_label,
        "schedule": args.schedule,
        "prototype": ("external-hip-diagnostic" if diagnostic_hip else
                      ("matched-bounded" if args.schedule == "matched" else "pipeline"))
                     if implementation_label != "handwritten" else None,
        "reference_library": str(args.reference_library) if args.reference_library else None,
        "diagnostic_hip": ({
            "source_path": str(args.diagnostic_hip),
            "kernel_symbol": state.diagnostic_symbol,
            "source_sha256": state.diagnostic_sha256,
            "abi": "nine-argument IPC (a, b, partial, ready, out, remote_partial, remote_ready, rank, world_size)",
            "launch": {"grid_x": args.total_blocks, "threads_per_block": THREADS,
                       "dynamic_shared_bytes": 64 * 1024},
        } if diagnostic_hip else None),
        "shape": {"m": args.m, "n": args.n, "k": args.k, "pe_count": PE_COUNT},
        "rank": state.rank,
        "input_mode": args.input_mode,
        "grid": grid,
        "flags": flags,
        "allocations": allocation_note,
        "residency": residency,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timing": {
            "metric": "hipEventElapsedTime around one kernel launch only",
            "samples_ms": samples_ms,
            "local_median_ms": float(statistics.median(samples_ms)),
            "local_p10_ms": _percentile(samples_ms, 10),
            "local_p90_ms": _percentile(samples_ms, 90),
            "wall_harness_complete_round_ms": wall_samples_ms,
            "excludes": ["allocation", "input copies", "compile", "ready/flag reset", "host cross-PE sync"],
            "cross_rank_aggregation": "elementwise max(samples_ms) before median/p10/p90; never mean four ranks",
            "wall_note": "optional harness wall includes host event polling plus device/context completion synchronization; use HIP-event samples for kernel comparison",
        },
        "verification": {"pre_timing": precheck, "post_timing": postcheck},
        "notes": [
            ("This is an explicit external HIP diagnostic artifact, not TileLang-generated code; "
             "the source path, symbol, and SHA-256 identify the measured variant."
             if diagnostic_hip else
             ("The matched candidate has a fixed fully-resident compute/communication split and is not M3."
              if args.schedule == "matched" else
              "The TileLang pipeline prototype uses one 512-thread CTA per PE and serially traverses tiles.")),
            "It is a correctness/source experiment, not a high-performance persistent implementation.",
        ],
    }
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"gemm-rs-benchmark rank={state.rank} samples={samples_ms} json={output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("tilelang", "handwritten"), default="tilelang")
    parser.add_argument("--schedule", choices=("single", "matched"), default="single",
                        help="TileLang schedule: preserve single-CTA baseline or use bounded matched roles")
    parser.add_argument("--reference-library", type=Path,
                        help="absolute path to libfused_gemm_rs_m*_n*_k*_ipc.so for handwritten")
    parser.add_argument("--comm-blocks", type=int, default=32,
                        help="matched/handwritten communication CTAs (default: 32)")
    parser.add_argument("--total-blocks", type=int, default=80,
                        help="matched candidate's fixed total grid CTAs (default: 80)")
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--input-mode", choices=("dyadic", "continuous"), default="dyadic",
                        help="input distribution; dyadic retains bitwise full-oracle assertions")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--wall-clock", action="store_true",
                        help="also store harness wall time (prepare + event polling + completion sync), not a kernel metric")
    parser.add_argument("--json-dir", type=Path, default=HERE / "benchmark_results")
    parser.add_argument("--json-prefix", default="gemm-rs")
    parser.add_argument("--export-hip", type=Path,
                        help="optional directory for per-rank TileLang generated HIP source")
    parser.add_argument("--diagnostic-hip", type=Path,
                        help="opt-in external exported matched HIP source; compiles only a diagnostic C-ABI launcher")
    parser.add_argument("--failure-dump-dir", type=Path,
                        help="on strict partial or reduce-scatter failure, save rank-local .npy/.json under build-old only")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
