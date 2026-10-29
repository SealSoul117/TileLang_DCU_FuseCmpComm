"""MPI/DCU runner for generalized fusion examples.

On an allocated eight-DCU node, from the old repository after sourcing
``scripts/dev_old/env.sh``, run with one MPI process per PE, for example::

    mpirun -n 8 --oversubscribe --bind-to none python -m \
        examples.distributed.fusion_generalized.run --kind rs --pe 8 \
        --dtype bfloat16 --m 1031 --n 521 --k 259 --weight-layout nk \
        --json-dir build-old/generalized-rs

Packing, weight transpose/padding, and output crop are TileLang kernels; host
arrays retain their user-facing [M,K] and default [N,K] layouts.

``--kind`` selects rs (GEMM + ReduceScatter), ar (GEMM + AllReduce), or ag
(AllGather + GEMM).  For AG, M is the gathered row count and N is this PE's
local output column count; PE r supplies rows [M*r//P, M*(r+1)//P) of A.
For AR/RS, each PE supplies its own complete [M,K] local GEMM contribution.
RS assigns the same contiguous logical row intervals to its output owners;
its compact storage reserves ceil(M/P) rows, with unused final rows zeroed.

Both IPC optimization passes are enabled by default.  The fused core is
specialized to single-node gfx936, 80 CTAs, 512 threads, and four GEMM pipeline
stages; --comm-blocks changes the compute/communication split.  Rectangular
tails are handled outside that aligned core, not by changing native GEMM.
JSON records keep prepare_ms, fused_ms, and crop_ms separate; upload, flag
reset, MPI barriers, and validation are outside these HIP-event intervals.
For multi-PE timing, take the elementwise maximum of samples across ranks
before computing a median.  This example validates with CPU Torch/NumPy;
Torch is not the GPU compute or communication backend.
"""

import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch
import tilelang
from tilelang.distributed.ipc_binding import bind_ipc
from tilelang.distributed.ipc_runtime import IPCContext

from .allgather_gemm import allgather_gemm
from .contract import FusionPlan, ceildiv, make_plan
from .gemm_allreduce import gemm_allreduce
from .gemm_reduce_scatter import gemm_reduce_scatter
from .prepare import crop_output, pack_ag_a_shard, pack_ar_rs_a, pack_weight


ROOT = Path(__file__).resolve().parents[3]
OLD_RS = ROOT / "examples" / "distributed" / "gemm_rs_prototype"
if str(OLD_RS) not in sys.path:
    sys.path.insert(0, str(OLD_RS))
import run_gemm_reduce_scatter as old_runner
from benchmark_gemm_reduce_scatter import EventPair, TileLangRun, _configure_hip, _mpi_abort
from gemm_reduce_scatter_conservative import _check_adapter_init


def _torch_dtype(dtype: str):
    return torch.bfloat16 if dtype == "bfloat16" else torch.float16


def encode(values, dtype: str) -> np.ndarray:
    """Return raw fp16/bf16 storage bits without ever reinterpreting BF16 as FP16."""
    return torch.from_numpy(np.asarray(values, dtype=np.float32)).to(_torch_dtype(dtype)).view(torch.uint16).numpy().copy()


def prepare_inputs(A, weight, plan: FusionPlan, *, rank: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Validate user-facing host inputs; GPU kernels perform all packing/crop work."""
    if np.asarray(A).dtype == np.uint16 or np.asarray(weight).dtype == np.uint16:
        raise TypeError("prepare_inputs accepts numeric values, not raw uint16 payload bits")
    if plan.kind == "ag" and (rank is None or not 0 <= rank < plan.world_size):
        raise ValueError("AG prepare_inputs requires a valid local rank")
    a = np.asarray(A, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32)
    raw_rows = (plan.logical_rows_for_rank(rank)[1] - plan.logical_rows_for_rank(rank)[0]
                if plan.kind == "ag" and rank is not None else
                (max(1, ceildiv(plan.m, plan.world_size)) if plan.kind == "ag" else plan.m))
    expected_a = (raw_rows, plan.k) if plan.kind == "ag" else (plan.m, plan.k)
    if a.shape != expected_a:
        raise ValueError(f"A must have shape {expected_a}, got {a.shape}")
    if w.shape != plan.logical_weight_shape:
        raise ValueError(f"weight must have shape {plan.logical_weight_shape}, got {w.shape}")
    return encode(a, plan.dtype), encode(w, plan.dtype)


def _dyadic_inputs(rank: int, epoch: int, plan: FusionPlan) -> tuple[np.ndarray, np.ndarray]:
    """Small bounded dyadic values avoid FP16 overflow while varying every epoch/rank."""
    rows = (plan.logical_rows_for_rank(rank)[1] - plan.logical_rows_for_rank(rank)[0]
            if plan.kind == "ag" else plan.m)
    ar = ((np.arange(rows)[:, None] + 3 * np.arange(plan.k)[None, :] + rank + epoch) % 4 - 2) / 16
    if plan.weight_layout == "nk":
        wr = ((5 * np.arange(plan.n)[:, None] + np.arange(plan.k)[None, :] + 2 * epoch + rank) % 4 - 2) / 16
    else:
        wr = ((np.arange(plan.k)[:, None] + 5 * np.arange(plan.n)[None, :] + 2 * epoch + rank) % 4 - 2) / 16
    return ar.astype(np.float32), wr.astype(np.float32)


def _dyadic_oracle(rank: int, epoch: int, plan: FusionPlan) -> np.ndarray:
    """Exact periodic-K oracle without a large host dense matmul.

    Four residue classes cover K, so every local FP32 GEMM is formed from four
    outer products, rounded once to payload precision, then AR/RS sums ranks
    in increasing order and rounds once again.  This mirrors the DSL contract
    while keeping verification practical for large rectangular examples.
    """
    rows = max(1, ceildiv(plan.m, plan.world_size)) if plan.kind == "rs" else plan.m
    row_ids = np.arange(plan.m, dtype=np.int64)[:, None]
    col_ids = np.arange(plan.n, dtype=np.int64)[None, :]
    counts = np.array([(plan.k + 3 - residue) // 4 for residue in range(4)], dtype=np.float32)

    def local(peer: int, ag: bool = False):
        result = np.zeros((plan.m, plan.n), dtype=np.float32)
        for residue in range(4):
            if ag:
                owners = ((row_ids + 1) * plan.world_size - 1) // plan.m
                local_rows = row_ids - (plan.m * owners // plan.world_size)
                a = ((local_rows + 3 * residue + owners + epoch) % 4 - 2) / 16
                b = ((5 * col_ids + residue + 2 * epoch + peer) % 4 - 2) / 16
            else:
                a = ((row_ids + 3 * residue + peer + epoch) % 4 - 2) / 16
                b = ((5 * col_ids + residue + 2 * epoch + peer) % 4 - 2) / 16
            result += counts[residue] * a * b
        return torch.from_numpy(result).to(_torch_dtype(plan.dtype)).float().numpy()

    if plan.kind == "ag":
        return encode(local(rank, ag=True), plan.dtype)
    reduced = np.zeros((plan.m, plan.n), dtype=np.float32)
    for peer in range(plan.world_size):
        reduced += local(peer)
    rounded = encode(reduced, plan.dtype)
    if plan.kind == "rs":
        begin, end = plan.logical_rows_for_rank(rank)
        padded = np.zeros((rows, plan.n), dtype=np.uint16)
        padded[:end - begin] = rounded[begin:end]
        return padded
    return rounded


def _local_partial_bits(rank: int, epoch: int, plan: FusionPlan) -> np.ndarray:
    """Physical tile-major local GEMM oracle for AR/RS workspace validation."""
    row_ids = np.arange(plan.m, dtype=np.int64)[:, None]
    col_ids = np.arange(plan.n, dtype=np.int64)[None, :]
    result = np.zeros((plan.m, plan.n), dtype=np.float32)
    for residue in range(4):
        count = (plan.k + 3 - residue) // 4
        a = ((row_ids + 3 * residue + rank + epoch) % 4 - 2) / 16
        b = ((5 * col_ids + residue + 2 * epoch + rank) % 4 - 2) / 16
        result += count * a * b
    logical = torch.from_numpy(result).to(_torch_dtype(plan.dtype)).float().numpy()
    physical = np.zeros((plan.m_physical, plan.n_physical), dtype=np.float32)
    for row in range(plan.m):
        _, packed = plan.logical_row_to_physical(row)
        physical[packed, :plan.n] = logical[row]
    return encode(physical.reshape(plan.tile_rows, 256, plan.tile_cols, 256)
                  .transpose(0, 2, 1, 3).reshape(plan.partial_shape), plan.dtype)


def _gathered_a_bits(epoch: int, plan: FusionPlan) -> np.ndarray:
    """Physical AG gathered-A oracle, including all padding rows/columns."""
    physical = np.zeros(plan.ag_gathered_a_shape, dtype=np.float32)
    cols = np.arange(plan.k, dtype=np.int64)[None, :]
    for owner in range(plan.world_size):
        begin, end = plan.logical_rows_for_rank(owner)
        local = np.arange(end - begin, dtype=np.int64)[:, None]
        physical[owner * plan.rows_per_rank_physical:owner * plan.rows_per_rank_physical + end - begin, :plan.k] = (
            ((local + 3 * cols + owner + epoch) % 4 - 2) / 16
        )
    return encode(physical, plan.dtype)


def _bytes(shape, dtype: str) -> int:
    return int(np.prod(shape)) * (4 if dtype == "uint32" else 2)


def _fused_bindings(kind: str) -> dict[str, str]:
    bindings = {"ready": "ipc_view_ready"}
    if kind == "ag":
        bindings["A_shard"] = "ipc_view_a_shard"
    else:
        bindings["partial"] = "ipc_view_partial"
    if kind == "ar":
        bindings["out"] = "ipc_view_out"
    return bindings


def _compile_fused(plan: FusionPlan, *, vector: bool, hoist: bool):
    if vector and not hoist:
        raise ValueError("--scalar is required with --no-hoist: vectorization depends on pointer hoisting")
    generator = {"ag": allgather_gemm, "rs": gemm_reduce_scatter, "ar": gemm_allreduce}[plan.kind]
    bound = bind_ipc(generator(plan), _fused_bindings(plan.kind))
    return tilelang.compile(
        bound, target="hip -mcpu=gfx936", execution_backend="ctypes",
        pass_configs={"tl.enable_dcu_ipc_hoist": hoist, "tl.enable_dcu_ipc_vectorize": vector},
    )


def _prebind_call(kernel, allocations, descriptor_names, rank: int, world: int, aliases=None):
    """Resolve TIR names and DeviceViews before events; return a fixed ABI call."""
    prim = next(f for f in kernel.adapter.ir_module.functions.values() if hasattr(f, "params"))
    values, ctypes_types = [], []
    for param in prim.params:
        buffer = prim.buffer_map.get(param)
        name = str(buffer.name) if buffer is not None else str(param.name)
        allocation_name = (aliases or {}).get(name, name)
        if allocation_name in allocations:
            values.append(C.c_void_p(allocations[allocation_name].data_ptr())); ctypes_types.append(C.c_void_p)
        elif name in descriptor_names:
            values.append(C.c_void_p(allocations[descriptor_names[name]].device_view().data_ptr())); ctypes_types.append(C.c_void_p)
        elif name in ("__tl_ipc_rank", "rank"):
            values.append(C.c_int32(rank)); ctypes_types.append(C.c_int32)
        elif name == "__tl_ipc_world_size":
            values.append(C.c_int32(world)); ctypes_types.append(C.c_int32)
        else:
            raise RuntimeError(f"unbound generated parameter {name}")
    values.append(C.c_void_p(0)); ctypes_types.append(C.c_void_p)
    kernel.adapter.lib.call.argtypes = ctypes_types
    kernel.adapter.lib.call.restype = C.c_int
    return lambda: old_runner._check(kernel.adapter.lib.call(*values), "adapter.lib.call")


def _require_resident(kernel, hip):
    state = type("State", (), {"hip": hip, "args": type("Args", (), {"total_blocks": 80})()})()
    state._require_full_residency_function = lambda function: TileLangRun._require_full_residency_function(state, function)
    TileLangRun._require_full_residency(state, kernel)


def _check_export(source: str, kind: str, vector: bool, peer_count: int) -> None:
    if not vector:
        return
    marker = {"ag": "ipc_ag_half8_i", "rs": "ipc_half8_i", "ar": "ipc_ar_half8_i"}[kind]
    descriptors = [("ipc_view_a_shard", 1)] if kind == "ag" else [("ipc_view_partial", peer_count)]
    if kind == "ar": descriptors.append(("ipc_view_out", peer_count))
    payload_positions = []
    for descriptor, expected in descriptors:
        positions = [index for index, line in enumerate(source.splitlines())
                     if f"tl::ipc::peer<std::byte>({descriptor}" in line]
        if len(positions) != expected:
            raise RuntimeError(f"IPC vectorization has {len(positions)} {descriptor} pointers, expected {expected}")
        payload_positions.extend(positions)
    if marker not in source or "uint4" not in source:
        raise RuntimeError("requested IPC vectorization did not reach generated HIP")
    marker_line = next(index for index, line in enumerate(source.splitlines()) if marker in line)
    if max(payload_positions) >= marker_line:
        raise RuntimeError("IPC peer address was not hoisted outside the vector loop")


def _allgather_config_hash(mpi, plan: FusionPlan, args) -> None:
    """Fail before allocation unless all ranks chose identical specialization."""
    canonical = json.dumps({"kind": plan.kind, "m": plan.m, "n": plan.n, "k": plan.k,
                            "pe": plan.world_size, "dtype": plan.dtype,
                            "weight_layout": plan.weight_layout, "comm_blocks": plan.comm_blocks,
                            "scalar": args.scalar, "no_hoist": args.no_hoist,
                            "warmup": args.warmup, "repeats": args.repeats,
                            "no_verify": args.no_verify}, sort_keys=True).encode()
    digest = hashlib.sha256(canonical).digest()
    byte = C.c_void_p(C.addressof(C.c_char.in_dll(mpi, "ompi_mpi_byte")))
    comm = C.c_void_p(C.addressof(C.c_char.in_dll(mpi, "ompi_mpi_comm_world")))
    recv = (C.c_ubyte * (32 * plan.world_size))()
    mpi.MPI_Allgather.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_void_p, C.c_int, C.c_void_p, C.c_void_p]
    mpi.MPI_Allgather.restype = C.c_int
    send = (C.c_ubyte * 32).from_buffer_copy(digest)
    old_runner._check(mpi.MPI_Allgather(send, 32, byte, recv, 32, byte, comm), "MPI_Allgather(config)")
    if any(bytes(recv[32 * peer:32 * (peer + 1)]) != digest for peer in range(plan.world_size)):
        raise RuntimeError("MPI ranks selected different generalized fusion configurations")


def run(args) -> None:
    plan = make_plan(args.kind, args.m, args.n, args.k, world_size=args.pe, dtype=args.dtype,
                     weight_layout=args.weight_layout, comm_blocks=args.comm_blocks)
    mpi = C.CDLL("libmpi.so")
    provided = C.c_int()
    old_runner._check(mpi.MPI_Init_thread(None, None, 1, C.byref(provided)), "MPI_Init_thread")
    ctx, allocations, launched, reader_safe, completed = None, {}, False, False, False
    try:
        ctx = IPCContext()
        if ctx.world_size != plan.world_size:
            raise RuntimeError(f"runtime world {ctx.world_size} does not match compiled PE count {plan.world_size}")
        module_path = Path(tilelang.__file__).resolve()
        if ROOT not in module_path.parents or "TileLang_DCU_develop" in str(module_path):
            raise RuntimeError(f"runner requires TileLang_DCU_old, loaded {module_path}")
        _allgather_config_hash(mpi, plan, args)
        rank, hip = ctx.rank, _configure_hip()
        shapes = {"B": plan.packed_weight_shape, "ready": (plan.flag_words,),
                  "out": plan.rs_output_shape if plan.kind == "rs" else plan.ar_output_shape,
                  "cropped": ((max(1, ceildiv(plan.m, plan.world_size)) if plan.kind == "rs" else plan.m), plan.n),
                  "weight_raw": plan.logical_weight_shape}
        if plan.kind == "ag":
            shapes.update({"A_raw": (max(1, ceildiv(plan.m, plan.world_size)), plan.k),
                           "A_shard": plan.ag_shard_shape, "A": plan.ag_gathered_a_shape})
        else:
            shapes.update({"A_raw": (plan.m, plan.k), "A": plan.ar_rs_packed_a_shape,
                           "partial": plan.partial_shape})
        for name, shape in shapes.items():
            fine = name in ("ready", "partial", "A_shard") or (name == "out" and plan.kind == "ar")
            allocations[name] = ctx.alloc(f"fusion-generalized-{name}", _bytes(shape, "uint32" if name == "ready" else plan.dtype), fine_grained=fine)
        # All four kernels share the same generated-source cache behavior; disable
        # it before the first preparation compile, not only before the fused one.
        tilelang.disable_cache()
        prep_a = tilelang.compile(pack_ag_a_shard(plan) if plan.kind == "ag" else pack_ar_rs_a(plan),
                                  target="hip -mcpu=gfx936", execution_backend="ctypes")
        prep_b = tilelang.compile(pack_weight(plan), target="hip -mcpu=gfx936", execution_backend="ctypes")
        crop = tilelang.compile(crop_output(plan), target="hip -mcpu=gfx936", execution_backend="ctypes")
        fused = _compile_fused(plan, vector=not args.scalar, hoist=not args.no_hoist)
        for kernel in (prep_a, prep_b, crop, fused):
            _check_adapter_init(kernel)
        _require_resident(fused, hip)
        source = fused.get_kernel_source()
        _check_export(source, plan.kind, not args.scalar, plan.world_size)
        if args.export_hip:
            args.export_hip.mkdir(parents=True, exist_ok=True)
            (args.export_hip / f"{plan.kind}-p{plan.world_size}-{plan.dtype}-rank{rank}.hip").write_text(source)
            for name, kernel in (("prepare-a", prep_a), ("prepare-b", prep_b), ("crop", crop)):
                (args.export_hip / f"{name}-p{plan.world_size}-{plan.dtype}-rank{rank}.hip").write_text(
                    kernel.get_kernel_source())
        call_prep_a = _prebind_call(prep_a, allocations, {}, rank, plan.world_size,
                                    {"A_packed": "A", "A_shard": "A_shard"})
        call_prep_b = _prebind_call(prep_b, allocations, {}, rank, plan.world_size, {"B_packed": "B"})
        call_fused = _prebind_call(fused, allocations,
                                   {value: key for key, value in _fused_bindings(plan.kind).items()},
                                   rank, plan.world_size)
        call_crop = _prebind_call(crop, allocations, {}, rank, plan.world_size,
                                  {"physical": "out", "output": "cropped"})
        event = EventPair(hip)
        prepare_ms, fused_ms, crop_ms = [], [], []
        verification = []
        verify_epochs = 2
        for epoch in range(verify_epochs + args.warmup + args.repeats):
            raw_a, raw_w = prepare_inputs(*_dyadic_inputs(rank, epoch, plan), plan, rank=rank)
            # AG's user input has its true floor-shard length; the symmetric
            # staging allocation is max-sized only to keep all PE allocations equal.
            if plan.kind == "ag":
                old_runner._check(hip.hipMemset(C.c_void_p(allocations["A_raw"].data_ptr()), 0,
                                                _bytes(shapes["A_raw"], plan.dtype)), "A_raw padding reset")
            old_runner._copy(hip, allocations["A_raw"].data_ptr(), raw_a.ctypes.data, raw_a.nbytes, 1)
            old_runner._copy(hip, allocations["weight_raw"].data_ptr(), raw_w.ctypes.data, raw_w.nbytes, 1)
            old_runner._check(hip.hipMemset(C.c_void_p(allocations["ready"].data_ptr()), 0, _bytes(shapes["ready"], "uint32")), "ready reset")
            for name in ("out", "cropped", "partial" if plan.kind != "ag" else "A"):
                old_runner._check(hip.hipMemset(C.c_void_p(allocations[name].data_ptr()), 0xA5, _bytes(shapes[name], plan.dtype)), f"poison {name}")
            prep = event.time_one(lambda: (
                call_prep_a(), call_prep_b(),
            ), 60)
            old_runner._check(hip.hipDeviceSynchronize(), "prepare completion"); ctx.sync()
            reader_safe = False
            launched = True
            fused_time = event.time_one(
                call_fused,
                60,
            )
            old_runner._check(hip.hipDeviceSynchronize(), "fused completion"); ctx.sync(); reader_safe = True
            cropped = event.time_one(call_crop, 60)
            old_runner._check(hip.hipDeviceSynchronize(), "crop completion")
            if not args.no_verify:
                expected = _dyadic_oracle(rank, epoch, plan)
                actual = np.empty(expected.shape, dtype=np.uint16)
                old_runner._copy(hip, actual.ctypes.data, allocations["cropped"].data_ptr(), actual.nbytes, 2)
                old_runner._check(hip.hipDeviceSynchronize(), "verification copy")
                bad = int(np.count_nonzero(actual != expected))
                if bad:
                    raise AssertionError(f"epoch {epoch}: {bad} payload-bit mismatches in cropped output")
                if plan.kind == "ag":
                    gathered = np.empty(plan.ag_gathered_a_shape, dtype=np.uint16)
                    old_runner._copy(hip, gathered.ctypes.data, allocations["A"].data_ptr(), gathered.nbytes, 2)
                    old_runner._check(hip.hipDeviceSynchronize(), "gather verification copy")
                    gathered_bad = int(np.count_nonzero(gathered != _gathered_a_bits(epoch, plan)))
                    if gathered_bad:
                        raise AssertionError(f"epoch {epoch}: {gathered_bad} gathered-A payload-bit mismatches")
                else:
                    partial = np.empty(plan.partial_shape, dtype=np.uint16)
                    old_runner._copy(hip, partial.ctypes.data, allocations["partial"].data_ptr(), partial.nbytes, 2)
                    old_runner._check(hip.hipDeviceSynchronize(), "partial verification copy")
                    partial_bad = int(np.count_nonzero(partial != _local_partial_bits(rank, epoch, plan)))
                    if partial_bad:
                        raise AssertionError(f"epoch {epoch}: {partial_bad} local-partial payload-bit mismatches")
                verification.append({"epoch": epoch, "output_bits": 0,
                                     "workspace_bits": 0, "status": "passed"})
                ctx.sync()  # all ranks finish host verification before reset/reuse
            if epoch >= verify_epochs + args.warmup:
                prepare_ms.append(prep); fused_ms.append(fused_time); crop_ms.append(cropped)
        event.close()
        record = {"rank": rank, "kind": plan.kind, "shape": [plan.m, plan.n, plan.k], "pe": plan.world_size,
                  "dtype": plan.dtype, "weight_layout": plan.weight_layout, "prepare_ms": prepare_ms,
                  "fused_ms": fused_ms, "crop_ms": crop_ms, "timing_scope": "HIP events; fused excludes prepare/crop/reset/sync"}
        record["verification"] = verification if not args.no_verify else {"status": "disabled"}
        record["tilelang_module"] = str(module_path)
        if args.json_dir:
            args.json_dir.mkdir(parents=True, exist_ok=True)
            (args.json_dir / f"{plan.kind}-rank{rank}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"RESULT rank={rank} kind={plan.kind} verification={record['verification']}", flush=True)
        completed = True
    except BaseException:
        traceback.print_exc()
        # Any one-rank compile, prepare, launch, or verification failure must
        # terminate the MPI world rather than strand peers in collectives.
        _mpi_abort(mpi)
        if ctx is not None and launched and not reader_safe:
            C.CDLL("libc.so.6")._exit(1)
        raise
    finally:
        if ctx is not None and completed:
            for allocation in reversed(list(allocations.values())):
                allocation.close()
            ctx.close()
            old_runner._check(mpi.MPI_Finalize(), "MPI_Finalize")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("ag", "rs", "ar"), required=True)
    parser.add_argument("--m", type=int, required=True); parser.add_argument("--n", type=int, required=True); parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--pe", type=int, choices=(4, 8), default=4)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--weight-layout", choices=("nk", "kn"), default="nk")
    parser.add_argument("--comm-blocks", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1); parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json-dir", type=Path); parser.add_argument("--export-hip", type=Path)
    parser.add_argument("--scalar", action="store_true", help="disable the verified IPC vectorization pass")
    parser.add_argument("--no-hoist", action="store_true", help="diagnostic scalar-only lowering without pointer hoisting")
    parser.add_argument("--no-verify", action="store_true", help="skip the two-epoch periodic-K bitwise oracle")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats <= 0: parser.error("--warmup must be nonnegative and --repeats positive")
    run(args)


if __name__ == "__main__":
    main()
