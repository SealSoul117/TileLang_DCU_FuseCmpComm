#!/usr/bin/env python3
"""Four-DCU PyTorch baseline: local GEMM, then blocking AllReduce."""

import argparse
import os
import socket
import time

import torch
import torch.distributed as dist


def timed_sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--local-k", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--backend", default="nccl")
    parser.add_argument(
        "--reduce-dtype", choices=("float16", "float32"), default="float32")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"This benchmark needs a DCU-enabled PyTorch; found torch {torch.__version__} "
            "with torch.cuda.is_available() == False")

    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=args.backend, init_method="env://")
    if world_size != 4:
        raise RuntimeError(f"Expected four processes, got {world_size}")

    device = torch.device("cuda", local_rank)
    a = torch.full((args.m, args.local_k), 0.03125,
                   dtype=torch.float16, device=device)
    # B is stored as [N, K], matching the TileLang example.
    b = torch.full((args.n, args.local_k), 0.0625,
                   dtype=torch.float16, device=device)
    reduce_dtype = getattr(torch, args.reduce_dtype)

    def gemm_then_allreduce():
        partial = torch.mm(a, b.t())
        if partial.dtype != reduce_dtype:
            partial = partial.to(reduce_dtype)
        dist.all_reduce(partial, op=dist.ReduceOp.SUM)
        return partial

    for _ in range(args.warmup):
        result = gemm_then_allreduce()
    torch.cuda.synchronize()
    dist.barrier()

    gemm_seconds = 0.0
    allreduce_seconds = 0.0
    for _ in range(args.iterations):
        start = timed_sync()
        result = torch.mm(a, b.t())
        if result.dtype != reduce_dtype:
            result = result.to(reduce_dtype)
        after_gemm = timed_sync()
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        after_allreduce = timed_sync()
        gemm_seconds += after_gemm - start
        allreduce_seconds += after_allreduce - after_gemm

    expected = world_size * args.local_k * 0.03125 * 0.0625
    max_error = (result.float() - expected).abs().max()
    metrics = torch.tensor(
        [1000.0 * gemm_seconds / args.iterations,
         1000.0 * allreduce_seconds / args.iterations,
         float(max_error)],
        dtype=torch.float64, device=device)
    # Report the slowest rank, which determines end-to-end latency.
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
    if rank == 0:
        gemm_ms, allreduce_ms, global_error = metrics.tolist()
        print(
            f"PyTorch serial: GEMM+cast={gemm_ms:.3f} ms, "
            f"AllReduce={allreduce_ms:.3f} ms, "
            f"total={gemm_ms + allreduce_ms:.3f} ms, "
            f"reduce_dtype={args.reduce_dtype}, max_error={global_error:g}, "
            f"warmup={args.warmup}, iterations={args.iterations}, "
            f"device={torch.cuda.get_device_name(local_rank)}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
