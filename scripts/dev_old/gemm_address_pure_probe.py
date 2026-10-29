"""Read-only diagnostic of upstream GEMM under the selected template view.

No production source is changed. Use TL_TEMPLATE_PATH to select the normal
native-lane template or the isolated raw-lane diagnostic template.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import tilelang
import tilelang.language as T


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/distributed/gemm_rs_prototype"))
import python_schedule_probe as runner


def interleaved_compute(m, n, k):
    """Same 48-compute/80-grid owner-interleaved traversal, with no IPC."""
    tiles = (m // 256) * (n // 256)
    @T.prim_func
    def compute_only(a: T.Tensor((m, k), "float16"),
                     b: T.Tensor((k, n), "float16"),
                     c: T.Tensor((m, n), "float16")):
        with T.Kernel(80, threads=512) as block_idx:
            if block_idx < 48:
                a_shared = T.alloc_shared((256, 16), "float16")
                b_shared = T.alloc_shared((16, 256), "float16")
                c_local = T.alloc_fragment((256, 256), "float32")
                for job in T.serial(block_idx, tiles, 48):
                    tile = (job % 4) * (tiles // 4) + job // 4
                    tile_row = tile // (n // 256)
                    tile_col = tile % (n // 256)
                    T.clear(c_local)
                    for kk in T.Pipelined(k // 16, num_stages=4,
                                         enable_register_pipeline=True, enable_warp_divergence=True):
                        T.copy(a[tile_row * 256, kk * 16], a_shared, major="row_major", enable_async=True)
                        T.copy(b[kk * 16, tile_col * 256], b_shared, major="col_major", enable_async=True)
                        T.gemm(a_shared, b_shared, c_local, k_pack=1, transpose_C=True)
                    T.copy(c_local, c[tile_row * 256, tile_col * 256])
                    T.sync_threads()
    return compute_only


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--k", type=int, default=1024)
    p.add_argument("--variant", choices=("official", "flat", "serial-inner-alloc", "interleaved"), default="official")
    p.add_argument("--blocks", type=int, default=16)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--input-mode", choices=("dyadic", "continuous"), default="dyadic")
    p.add_argument("--source-only", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tilelang.disable_cache()
    if args.variant == "official":
        spec = importlib.util.spec_from_file_location("official_gemm_address_probe", ROOT / "examples/dcu/example_gemm.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        func = module.matmul.get_tir(args.m, args.n, args.k, 256, 256, 16)
    elif args.variant == "interleaved":
        func = interleaved_compute(args.m, args.n, args.k)
    else:
        func = runner.make_probe(args.variant, args.m, args.n, args.k, args.blocks)
    source = runner.export_source(func, args.output)
    template = Path(os.environ["TL_TEMPLATE_PATH"])
    report = {
        "variant": args.variant, "shape": [args.m, args.n, args.k],
        "input_mode": args.input_mode, "template": str(template),
        "helper_sha256": hashlib.sha256((template / "tl_templates/dcu_hip/intrin.h").read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "status": "SOURCE_ONLY",
    }
    try:
        if not args.source_only:
            kernel = runner._compile(func)
            (args.output / "compiled.hip").write_text(kernel.adapter.lib_code)
            runner.run_one(func, args, kernel=kernel)
            report["status"] = "PASS"
    except Exception as exc:
        report.update(status="FAIL", error=str(exc))
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
