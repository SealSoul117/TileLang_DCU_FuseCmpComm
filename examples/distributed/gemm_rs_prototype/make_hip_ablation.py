"""Explicit HIP-only ablations, NOT TileLang compiler output/implementation.

Start from an archived, numerically checked matched 80/48/32 generated kernel.
Keep its ABI, GEMM arithmetic, tile ownership, publication and completion intact.
Each cumulative stage is independently benchmarked by --diagnostic-hip.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"expected one anchor: {before[:100]}")
    return source.replace(before, after, 1)


def loop_end(source, start):
    opening = source.index("{", start)
    depth = 1
    cursor = opening + 1
    while depth:
        depth += (source[cursor] == "{") - (source[cursor] == "}")
        cursor += 1
    return cursor


def hoist(source):
    # Allocation descriptors and peer mappings are immutable for this launch.
    pattern = r"        void\* (remote_partial[0-3]_remote_ptr_\d+) = (tl::ipc::peer<std::byte>\(ipc_view_partial, [0-3], __tl_ipc_rank, __tl_ipc_world_size, 0\));"
    matches = list(re.finditer(pattern, source))
    if len(matches) != 4:
        raise ValueError("expected four per-element checked peer resolutions")
    declarations = "\n".join("    void* " + m[1] + " = " + m[2] + ";" for m in matches)
    result = re.sub(pattern, "", source)
    anchor = "  } else {\n    for (int _tmp_1"
    return replace_once(result, anchor, "  } else {\n" + declarations + "\n    for (int _tmp_1")


def vectorize(source, m, n):
    names = re.findall(r"void\* (remote_partial[0-3]_remote_ptr_\d+) =", source)
    if len(names) != 4:
        raise ValueError("hoisted pointer names not found")
    start = source.index("      for (int _tmp_2")
    end = loop_end(source, start)
    tiles_per_pe = (m // 256) * (n // 256) // 4
    pointer_lines = "\n".join(
        f"      const uint4* audit_src{p} = reinterpret_cast<const uint4*>({name}) + audit_tile * 8192;"
        for p, name in enumerate(names))
    # Unroll sources in C++, exactly retaining source order 0,1,2,3.
    replacement = f"""      const int audit_owned = int(blockIdx.x) - 48 + _tmp_1 * 32;
      const int audit_tile = __tl_ipc_rank * {tiles_per_pe} + audit_owned;
{pointer_lines}
      const uint4* audit_sources[4] = {{audit_src0, audit_src1, audit_src2, audit_src3}};
      uint4* audit_dest = reinterpret_cast<uint4*>(out +
          size_t(audit_owned / {n // 256}) * 256 * {n} + (audit_owned % {n // 256}) * 256);
      // Wave/lane mapping protects the historical max256/launch512 convention.
      const int audit_tid = tl::get_warp_idx_sync(64) * 64 + (int(threadIdx.x) & 63);
      for (int audit_i = audit_tid; audit_i < 8192; audit_i += 512) {{
        float audit_sums[8] = {{}};
        #pragma unroll
        for (int p = 0; p < 4; ++p) {{
          uint4 word = audit_sources[p][audit_i];
          half_t values[8]; __builtin_memcpy(values, &word, 16);
          #pragma unroll
          for (int j = 0; j < 8; ++j) audit_sums[j] += float(values[j]);
        }}
        half_t values[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) values[j] = half_t(audit_sums[j]);
        uint4 word; __builtin_memcpy(&word, values, 16);
        audit_dest[(audit_i / 32) * ({n} / 8) + audit_i % 32] = word;
      }}"""
    return source[:start] + replacement + source[end:]


def cache_lane(source):
    result = source.replace("tl::get_lane_idx(64)", "audit_lane")
    return replace_once(result, "  float c_local[128];", "  const int audit_lane = tl::get_lane_idx(64);\n  float c_local[128];")


def scalar_simple(source, m, n):
    """Separate index/bounds simplification from vectorization of the loop."""
    names = re.findall(r"void\* (remote_partial[0-3]_remote_ptr_\d+) =", source)
    start = source.index("      for (int _tmp_2")
    end = loop_end(source, start)
    loads = "\n".join(f"        float audit_p{p} = float(reinterpret_cast<const half_t*>({name})[audit_tile * 65536 + audit_i]);"
                      for p, name in enumerate(names))
    replacement = f"""      const int audit_owned = int(blockIdx.x) - 48 + _tmp_1 * 32;
      const size_t audit_tile = __tl_ipc_rank * {m // 256 * (n // 256) // 4} + audit_owned;
      half_t* audit_dest = out + size_t(audit_owned / {n // 256}) * 256 * {n} + (audit_owned % {n // 256}) * 256;
      const int audit_tid = tl::get_warp_idx_sync(64) * 64 + (int(threadIdx.x) & 63);
      for (int audit_i = audit_tid; audit_i < 65536; audit_i += 512) {{
{loads}
        audit_dest[(audit_i / 256) * {n} + audit_i % 256] = half_t(((audit_p0 + audit_p1) + audit_p2) + audit_p3);
      }}"""
    return source[:start] + replacement + source[end:]


def reference_compute(source, reference, m, n):
    """A diagnostic upper-bound control: substitute the entire reference GEMM.

    Not a compiler fix or single-instruction intervention. Keeps the generated
    communication/publication code and host ABI, isolates the remaining GEMM gap.
    """
    ref = reference.read_text()
    start = ref.index("__device__ __forceinline__ void compute_tile(")
    helper = ref[start:loop_end(ref, start)]
    result = replace_once(source, 'extern "C" __global__', helper + '\n\nextern "C" __global__')
    begin = result.index("      #pragma unroll\n      for (int i = 0;")
    end = result.index("tl::ipc::fence_sys();", begin)
    tile_count = m // 256 * (n // 256)
    call = f"""      const int audit_job = int(blockIdx.x) + _tmp * 48;
      const int audit_tile = (audit_job % 4) * {tile_count // 4} + audit_job / 4;
      compute_tile(a, b, remote_partial3 + size_t(audit_tile) * 65536,
                   audit_tile % {n // 256}, audit_tile / {n // 256}, buf_dyn_shmem);
"""
    return result[:begin] + call + result[end:]


def hoist_ready(source, m, n):
    """Resolve checked flag pointers once, validate the entire used flag span."""
    words = m // 256 * (n // 256) * 4 + 4 + 32
    # Each store source line has a peer<uint>(..., byte_offset) enclosed in
    # reinterpret_cast<uint32_t*>(...). Parse parentheses instead of assuming
    # a particular simplifier's offset spelling.
    result = source
    starts = list(re.finditer(r"tl::ipc::peer<uint>\(", source))
    assert len(starts) == 8
    for match in reversed(starts):
        opening = match.end() - 1
        depth = 1
        cursor = opening + 1
        boundaries = [cursor]
        while depth:
            c = source[cursor]
            if c == "," and depth == 1:
                boundaries.append(cursor + 1)
            depth += (c == "(") - (c == ")")
            cursor += 1
        arguments = [source[a:b-1].strip() for a, b in zip(boundaries, boundaries[1:] + [cursor])]
        assert len(arguments) == 5 and arguments[0] == "ipc_view_ready"
        p = int(arguments[1])
        result = result[:match.start()] + f"(audit_ready{p} + ({arguments[4]}) / sizeof(uint))" + result[cursor:]
    result = re.sub(r"^\s*void\* remote_ready[0-3]_remote_ptr_\d+ = tl::ipc::peer<std::byte>\(ipc_view_ready, .*?;\n", "", result, flags=re.M)
    declarations = "".join(f"  uint* audit_ready{p} = tl::ipc::peer<uint>(ipc_view_ready, {p}, __tl_ipc_rank, __tl_ipc_world_size, {words-1} * sizeof(uint)) - {words-1};\n" for p in range(4))
    return replace_once(result, "  float c_local[128];", declarations + "  float c_local[128];")


def idxen(source, k):
    result = replace_once(source, "a_dcu_res = tl::make_wave_buffer_resource(a + (0))",
                          f"a_dcu_res = tl::make_wave_buffer_resource<{(k * 2).bit_length() - 1}>(a + (0))")
    lines = []
    count = 0
    for line in result.splitlines():
        if "tl::cp_async_gs<" in line and ", a_dcu_res, " in line:
            prefix, offset = line.split(", a_dcu_res, ", 1)
            assert offset.endswith(");")
            offset = offset[:-2]
            line = prefix.replace("tl::cp_async_gs<", "tl::cp_async_gs_idxen<") + f", a_dcu_res, ({offset}) % {k*2}, ({offset}) / {k*2});"
            count += 1
        lines.append(line)
    if count != 12:
        raise ValueError(f"expected 12 static A copies, got {count}")
    return "\n".join(lines) + "\n"


def tile_base(source, m, n, k):
    # Move resource construction from kernel entry to the current compute tile.
    lines = source.splitlines()
    resources = [line for line in lines if "_dcu_res = tl::make_wave_buffer_resource" in line]
    assert len(resources) == 2
    result = source
    for line in resources:
        result = replace_once(result, line + "\n", "")
    tile_count = m // 256 * (n // 256)
    anchor = re.search(r"    for \(int _tmp = .*?\{\n", result).group()
    declarations = f"""      const int audit_job = int(blockIdx.x) + _tmp * 48;
      const int audit_compute_tile = (audit_job % 4) * {tile_count // 4} + audit_job / 4;
      const size_t audit_a_base = size_t(audit_compute_tile / {n // 256}) * 256 * {k};
      const size_t audit_b_base = size_t(audit_compute_tile % {n // 256}) * 256;
"""
    for line in resources:
        declarations += line.replace("a + (0)", "a + audit_a_base").replace("b + (0)", "b + audit_b_base") + "\n"
    result = replace_once(result, anchor, anchor + declarations)
    # idxen's row index is allocation-relative until this stage; subtract tile row.
    result = re.sub(r"(, a_dcu_res, .*), (.*) / " + str(k*2) + r"\);",
                    lambda x: x[1] + ", (" + x[2] + f" - audit_a_base * 2) / {k*2});", result)
    result = re.sub(r"(, b_dcu_res, )(.*)\);",
                    lambda x: x[1] + "(" + x[2] + " - audit_b_base * 2));", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--reference", type=Path, help="optional read-only handwritten GEMM control")
    args = parser.parse_args()
    if args.m not in (2048, 4096) or args.n != args.m or args.k != args.m // 4:
        raise ValueError("only the two archived reference shapes are supported")
    source = args.source.read_text()
    stages = [("v0_generated", source)]
    stages.append(("v1_hoisted", hoist(stages[-1][1])))
    stages.append(("v2_vector", vectorize(stages[-1][1], args.m, args.n)))
    stages.append(("v3_lane_cached", cache_lane(stages[-1][1])))
    stages.append(("v4_idxen", idxen(stages[-1][1], args.k)))
    stages.append(("v5_tile_base", tile_base(stages[-1][1], args.m, args.n, args.k)))
    # Independent lane-only intervention detects effects hidden by communication.
    stages.append(("lane_only", cache_lane(source)))
    stages.append(("v1b_scalar_simple", scalar_simple(hoist(source), args.m, args.n)))
    if args.reference:
        stages.append(("reference_compute", reference_compute(stages[2][1], args.reference, args.m, args.n)))
        stages.append(("reference_compute_ready", hoist_ready(stages[-1][1], args.m, args.n)))
    stages.append(("v4_ready_hoisted", hoist_ready(stages[4][1], args.m, args.n)))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"diagnostic_only": True, "source": str(args.source.resolve()), "stages": {}}
    for name, content in stages:
        path = args.output / f"{name}.hip"
        path.write_text(content)
        manifest["stages"][name] = {"path": str(path.resolve()), "sha256": hashlib.sha256(content.encode()).hexdigest()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
