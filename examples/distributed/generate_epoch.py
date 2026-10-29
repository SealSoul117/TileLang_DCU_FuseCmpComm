"""Generate epoch fusion HIP from TileLang DSL without an application runtime.

Run as: python -m examples.distributed.generate_epoch --kind ar --output build/epoch-ar
The TileLang native libraries and the HIP toolchain must already be built/installed.
This compiles kernels but does not launch them or allocate an IPC workspace.
"""
import argparse
from pathlib import Path


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('ar', 'ag', 'rs'), required=True)
    parser.add_argument('--pe', type=int, choices=(4, 8), default=8)
    parser.add_argument('--m', type=int, nargs='+', default=[2048, 4096])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.kind != 'ar' and args.pe != 8:
        parser.error('The AG/RS epoch generation recipe currently supports --pe 8 only.')
    if any(m <= 0 for m in args.m) or len(set(args.m)) != len(args.m):
        parser.error('--m values must be positive and distinct.')
    if args.output.exists():
        parser.error('--output must be a new directory.')

    import tilelang
    from tilelang.distributed.ipc_binding import bind_ipc
    from .fusion_generalized.contract import make_plan

    if args.kind == 'ar':
        from .gemm_ar_epoch.gemm_allreduce_epoch import gemm_allreduce_epoch as generator
        bindings = {'ready': 'ipc_view_ready', 'partial': 'ipc_view_partial', 'out': 'ipc_view_out'}
    elif args.kind == 'ag':
        from .gemm_agrs_epoch.allgather_gemm_epoch import allgather_gemm_epoch as generator
        bindings = {'ready': 'ipc_view_ready', 'A_comm': 'ipc_view_a'}
    else:
        from .gemm_agrs_epoch.gemm_reduce_scatter_epoch import gemm_reduce_scatter_epoch as generator
        bindings = {'ready': 'ipc_view_ready', 'partial': 'ipc_view_partial'}
    n, k = (768, 2048) if args.kind == 'ag' else (2048, 512)
    args.output.mkdir(parents=True, exist_ok=False)
    tilelang.disable_cache()
    for m in args.m:
        plan = make_plan(args.kind, m, n, k, world_size=args.pe,
                         dtype='bfloat16', weight_layout='nk')
        kernel = tilelang.compile(
            bind_ipc(generator(plan), bindings), target='hip -mcpu=gfx936',
            execution_backend='ctypes',
            pass_configs={'tl.enable_dcu_ipc_hoist': True, 'tl.enable_dcu_ipc_vectorize': True},
        )
        source = kernel.get_kernel_source()
        if args.kind == 'ag':
            # Preserve the checks from the existing epoch generation recipe.
            prologue = source.split('tl::cp_async_wait<4>();', 1)[0]
            if prologue.count('tl::wave_barrier();') != 4:
                raise RuntimeError('AG GEMM prologue lost its four AB-pair barriers')
            if not any('*(uint4*)' in line and 'remote_dest_remote_ptr' in line
                       for line in source.splitlines()):
                raise RuntimeError('AG push did not generate 128-bit vector stores')
        else:
            _check_export(source, args.kind, True, args.pe)
        path = args.output / f'{args.kind}_m{m}_n{n}_k{k}_p{args.pe}.hip'
        path.write_text(source, encoding='utf-8')
        print(path, flush=True)


if __name__ == '__main__':
    main()
