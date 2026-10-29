"""Four-PE, same-allocation A00/B10/C11 ablation of the current fused examples.

Launch with mpirun -np 4. This uses the existing production generators, IPC
allocation/lifecycle and full numerical oracles; it never substitutes HIP.
"""
from __future__ import annotations
import argparse
import ctypes as ct
from functools import lru_cache
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for directory in ('gemm_rs_prototype', 'gemm_ar_prototype', 'ag_gemm_prototype'):
    sys.path.insert(0, str(HERE / directory))
import tilelang
from tilelang import tvm
import benchmark_gemm_reduce_scatter as rs
import benchmark_gemm_allreduce as ar
import benchmark_allgather_gemm as ag
import gemm_reduce_scatter_scheduled as rs_gen
import gemm_allreduce_scheduled as ar_gen
import allgather_gemm_scheduled as ag_gen

VARIANTS = {'A00': (False, False), 'B10': (True, False), 'C11': (True, True)}
MARKERS = {'rs': 'ipc_half8_i', 'ar': 'ipc_ar_half8_i', 'ag': 'ipc_ag_half8_i'}


def configs(label):
    hoist, vector = VARIANTS[label]
    return {'tl.enable_dcu_ipc_hoist': hoist, 'tl.enable_dcu_ipc_vectorize': vector}


def inspect_hip(source, kind, label):
    """Check actual JIT payload pointer placement, not merely config values."""
    marker = MARKERS[kind]
    assert (marker in source) == VARIANTS[label][1], (kind, label, 'vector marker')
    assert 'can_vectorize_half' not in source
    peer_pattern = (r'tl::ipc::peer<std::byte>\(ipc_view_a_shard' if kind == 'ag'
                    else r'tl::ipc::peer<std::byte>\(ipc_view_(?:partial|out)')
    pointers = [m.start() for m in re.finditer(peer_pattern, source)]
    assert len(pointers) == {'ag': 1, 'rs': 4, 'ar': 8}[kind], (kind, label, pointers)
    payload_name = 'remote_source' if kind == 'ag' else 'remote_partial0'
    candidates = []
    for match in re.finditer(r'for\s*\(', source):
        brace = source.index('{', match.end())
        depth, end = 1, brace + 1
        while depth:
            depth += (source[end] == '{') - (source[end] == '}')
            end += 1
        body = source[brace:end]
        if payload_name in body and not re.search(r'for\s*\(', body):
            candidates.append((match.start(), brace, end))
    assert len(candidates) == 1, (kind, label, 'ambiguous payload loop', candidates)
    start, brace, end = candidates[0]
    inside = sum(brace < p < end for p in pointers)
    assert inside == (len(pointers) if label == 'A00' else 0), (kind, label, inside)
    if label != 'A00':
        assert all(p < start for p in pointers), (kind, label, 'not before payload')
    # Check the actual 128-bit payload accesses, not just the loop marker.
    vector_access = r'\*\(uint4\*\)\(\(\(half_t\*\)(remote_(?:partial[0-3]|out[0-3]|source)_remote_ptr_\d+)'
    payload_vectors = re.findall(vector_access, source[brace:end])
    expected_vectors = {'rs': 4, 'ar': 8, 'ag': 1}[kind] if label == 'C11' else 0
    assert len(payload_vectors) == len(set(payload_vectors)) == expected_vectors, (kind, label, payload_vectors)
    return {'peer_bases': len(pointers), 'bases_inside_payload': inside,
            'vector_peer_accesses': len(payload_vectors),
            'half8': marker in source, 'sha256': hashlib.sha256(source.encode()).hexdigest()}


def prebind_rs(state, kernel):
    funcs = [f for f in kernel.adapter.ir_module.functions.values()
             if hasattr(f, 'buffer_map') and hasattr(f, 'params')]
    assert len(funcs) == 1
    f = funcs[0]
    names = tuple(str(f.buffer_map[p].name) if p in f.buffer_map else str(p.name) for p in f.params)
    assert names == ('A', 'B', 'partial', 'ready', 'out', 'ipc_view_partial',
                     'ipc_view_ready', '__tl_ipc_rank', '__tl_ipc_world_size'), names
    kernel.adapter.lib.call.argtypes = [ct.c_void_p] * 7 + [ct.c_int32, ct.c_int32, ct.c_void_p]
    kernel.adapter.lib.call.restype = ct.c_int
    allocations = (state.a, state.b, state.partial, state.ready, state.out,
                   state.partial.device_view(), state.ready.device_view())
    return tuple(ct.c_void_p(x.data_ptr()) for x in allocations) + (
        ct.c_int32(state.rank), ct.c_int32(4), ct.c_void_p(0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=('rs', 'ar', 'ag'), required=True)
    p.add_argument('--size', type=int, choices=(2048, 4096, 8192, 16384), required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--warmup', type=int, default=30)
    p.add_argument('--repeats', type=int, default=30)
    p.add_argument('--timeout-seconds', type=float, default=120)
    opt = p.parse_args()
    assert opt.warmup >= 0 and opt.repeats > 0
    assert Path(tilelang.__file__).resolve().is_relative_to(ROOT), tilelang.__file__
    assert Path(os.environ['TILELANG_DEV_BUILD_ROOT']).resolve() == ROOT / 'build-old'
    opt.output.mkdir(parents=True, exist_ok=True)
    s = opt.size
    m, n, k = (s, s // 4, s) if opt.kind == 'ag' else (s, s, s // 4)
    # Inputs do not change between variants/iterations. Reuse the *same full
    # CPU oracle* instead of repeating large CPU GEMMs for every pre/post
    # check; all device results are still copied and checked elementwise.
    rs.runner._partial_reference = lru_cache(maxsize=4)(rs.runner._partial_reference)
    ag._expected = lru_cache(maxsize=1)(ag._expected)
    generator = {'rs': rs_gen, 'ar': ar_gen, 'ag': ag_gen}[opt.kind]
    comm = ({2048: 32, 4096: 32, 8192: 8, 16384: 4}[s] if opt.kind == 'rs'
            else ar_gen.default_comm_blocks(m, n, k) if opt.kind == 'ar'
            else ag_gen.schedule_parameters(m, n, k)['comm_blocks'])
    args = argparse.Namespace(m=m, n=n, k=k, total_blocks=80, comm_blocks=comm,
        scalar=False, continuous=False, input_mode='dyadic', schedule='matched',
        implementation='tilelang', export_hip=None)
    _, bound = generator.build_bound(m, n, k, total_blocks=80, comm_blocks=comm)
    kernels, evidence, residencies = {}, {}, {}
    tilelang.disable_cache()

    def compile_one(state, label):
        # Distinct symbols prevent accidental interposition between three
        # simultaneously loaded ctypes launch libraries.
        func = bound.with_attr('global_symbol', f'fused_{opt.kind}_ablation_{label}')
        kernel = tilelang.compile(func, target='hip -mcpu=gfx936', execution_backend='ctypes',
                                  pass_configs=configs(label))
        rs._check_adapter_init(kernel)
        source = kernel.get_kernel_source()
        (opt.output / f'{label}-rank{state.rank}.hip').write_text(source)
        evidence[label] = inspect_hip(source, opt.kind, label)
        rs.TileLangRun._require_full_residency(state, kernel)
        residencies[label] = dict(state.residency)
        kernels[label] = kernel
        return kernel

    class RsRun(rs.TileLangRun):
        def _compile(self): return compile_one(self, 'C11')
    class ArRun(ar.Run):
        def _compile_tilelang(self): return compile_one(self, 'C11')
    class AgRun(ag.Run):
        def _compile(self): return compile_one(self, 'C11')

    state = events = None
    snapshots = []
    try:
        state = {'rs': RsRun, 'ar': ArRun, 'ag': AgRun}[opt.kind](args)
        for label in ('A00', 'B10'):
            compile_one(state, label)
        calls = {}
        for label, kernel in kernels.items():
            if opt.kind == 'rs':
                call_args = prebind_rs(state, kernel)
            elif opt.kind == 'ar':
                state.tile_kernel = kernel
                call_args = state._prebind_tile_args()
            else:
                state.kernel = kernel
                call_args = state._prebind_args()
            calls[label] = (kernel.adapter.lib.call, call_args)
        events = rs.EventPair(state.hip)
        (opt.output / f'pid-rank{state.rank}.json').write_text(json.dumps({'pid': os.getpid()}))
        state.ctx.sync()
        expected_pids = {str(json.loads((opt.output / f'pid-rank{r}.json').read_text())['pid']) for r in range(4)}

        def snapshot(tag):
            if state.rank == 0:
                active = {x.name for x in Path('/sys/class/kfd/kfd/proc').iterdir()}
                snapshots.append({'tag': tag, 'gpu_pids': sorted(active), 'foreign': sorted(active - expected_pids)})
                if active - expected_pids:
                    raise RuntimeError(f'external GPU processes during benchmark: {active - expected_pids}')

        def prepare(poison=False):
            if opt.kind == 'ag':
                state.prepare(poison=poison)
            else:
                if poison:
                    for allocation, count in ((state.partial, m * n * 2),
                            (state.out, m * n * 2 // (4 if opt.kind == 'rs' else 1))):
                        rs.runner._check(state.hip.hipMemset(ct.c_void_p(allocation.data_ptr()), 0xA5, count), 'poison')
                state.prepare()

        def launch(label):
            state.launched, state.reader_safe = True, False
            call, values = calls[label]
            rs.runner._check(call(*values), 'prebound launch ' + label)
            rs.runner._check(state.hip.hipGetLastError(), 'launch HIP error')

        def complete():
            if opt.kind == 'ag': state.complete_local()
            else: state.complete()

        def finish():
            if opt.kind != 'rs': state.finish_round()

        def check():
            if opt.kind == 'ag': state.check(0)
            else:
                module = ar if opt.kind == 'ar' else rs
                module._full_check(state.hip, state.partial_ptr, state.out_ptr,
                                   state.rank, m, n, k, 'dyadic')

        snapshot('precheck')
        for label in VARIANTS:
            prepare(True); launch(label); complete(); check(); finish()
        # All six permutations balance both position and predecessor.  This
        # is a single allocation set, not separately initialized process runs.
        orders = list(itertools.permutations(VARIANTS))
        samples = {label: [] for label in VARIANTS}
        for iteration in range(-opt.warmup, opt.repeats):
            if iteration % 6 == 0: snapshot(f'iteration-{iteration}')
            for label in orders[iteration % len(orders)]:
                prepare()
                elapsed = events.time_one(lambda label=label: launch(label), opt.timeout_seconds)
                complete(); finish()
                if iteration >= 0: samples[label].append(elapsed)
        for label in VARIANTS:
            prepare(True); launch(label); complete(); check(); finish()
        snapshot('postcheck')
        record = {'kind': opt.kind, 'shape': [m,n,k], 'rank': state.rank,
            'job': os.environ.get('SLURM_JOB_ID'), 'compute_blocks': 80-comm, 'comm_blocks': comm,
            'tilelang_path': tilelang.__file__, 'build_root': os.environ['TILELANG_DEV_BUILD_ROOT'],
            'warmup': opt.warmup, 'repeats': opt.repeats, 'samples_ms': samples,
            'configs': {label: configs(label) for label in VARIANTS}, 'jit_evidence': evidence,
            'residency': residencies, 'gpu_process_snapshots': snapshots,
            'verification': 'A/B/C full dyadic intermediate+output pre/post poison checks passed',
            'metric': 'aggregate elementwise max over four ranks, then median; GPU events only',
            'source': 'unmodified current TileLang generators; real JIT; identical allocations/inputs'}
        (opt.output / f'rank{state.rank}.json').write_text(json.dumps(record, indent=2))
        print(f'ABLATION_PASS {opt.kind} {s} rank={state.rank}', flush=True)
    except BaseException as error:
        print(f'ABLATION_FAILED {opt.kind} {s}: {error}', flush=True)
        if getattr(error, 'stderr', None):
            print(error.stderr, flush=True)
        rs._mpi_abort(state.mpi if state is not None else ct.CDLL('libmpi.so'))
    finally:
        if events is not None: events.close()
        if state is not None: state.close()


if __name__ == '__main__':
    main()
