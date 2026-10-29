#!/usr/bin/env python3
"""Apply one strict HIP-only tile-base alignment diagnostic.

This tool consumes an exported *uppercase-A/B* matched GEMM+ReduceScatter HIP
artifact.  It is not TileLang output and does not modify TileLang, templates,
or the benchmark runner.  Its sole intervention is to align the already
idxen-enabled A/B resource bases with the owner-interleaved current tile:

* move the two entry resource declarations into the compute owner-tile loop;
* make A's resource start at ``tile_row * 256 * K`` and either subtract
  ``tile_row * 256`` from every A idxen row (default), or use ``--local-row``
  after an exhaustive proof that the source row equals tile-row plus that
  native local row expression;
* make B's resource start at ``tile_col * 256`` and subtract
  ``tile_col * 256 * 2`` from every B flat byte offset.

All communication, scheduling, arithmetic, barriers, and every other source
byte remain from the supplied HIP artifact.  Narrow anchors and exact count
checks intentionally reject unexpected code-generation shapes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


TILE = 256
COMPUTE_BLOCKS = 48


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _replace_once(source: str, before: str, after: str, label: str) -> str:
    count = source.count(before)
    if count != 1:
        raise ValueError(f"{label}: expected exactly one anchor, found {count}: {before}")
    return source.replace(before, after, 1)


def _matching_paren(source: str, opening: int) -> int:
    if source[opening] != "(":
        raise ValueError("internal parser expected an opening parenthesis")
    depth = 1
    index = opening + 1
    while depth:
        if index >= len(source):
            raise ValueError("unterminated HIP call")
        char = source[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        index += 1
    return index - 1


def _split_top_level_arguments(text: str) -> list[str]:
    arguments: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            arguments.append(text[start:index])
            start = index + 1
    arguments.append(text[start:])
    return arguments


def _calls_with_resource(source: str, function_prefix: str, resource: str,
                         expected_arguments: int) -> list[tuple[int, int, str, list[str]]]:
    """Return complete calls whose top-level args reference one resource var."""
    calls: list[tuple[int, int, str, list[str]]] = []
    start = 0
    while True:
        found = source.find(function_prefix, start)
        if found < 0:
            break
        opening = source.find("(", found + len(function_prefix))
        if opening < 0:
            raise ValueError(f"missing call parenthesis after {function_prefix}")
        closing = _matching_paren(source, opening)
        call = source[found:closing + 1]
        arguments = _split_top_level_arguments(source[opening + 1:closing])
        if len(arguments) != expected_arguments:
            raise ValueError(f"unexpected argument count in {function_prefix}: {len(arguments)}")
        if arguments[1].strip() == resource:
            calls.append((found, closing + 1, call, arguments))
        start = closing + 1
    return calls


def _replace_calls(source: str, calls: list[tuple[int, int, str, list[str]]], replacements: list[str]) -> str:
    if len(calls) != len(replacements):
        raise ValueError("internal call replacement count mismatch")
    for (start, end, _, _), replacement in reversed(list(zip(calls, replacements))):
        source = source[:start] + replacement + source[end:]
    return source


def _validate_shape(m: int, n: int, k: int) -> None:
    if (m, n, k) not in ((2048, 2048, 512), (4096, 4096, 1024)):
        raise ValueError("supported shapes are exactly 2048x2048x512 and 4096x4096x1024")


def _source_row_anchor(k: int) -> str:
    """The exact generated global A-row decomposition accepted by this audit."""
    row_tile_shift = 2 if k == 1024 else 1
    return (
        f"((((((((int)blockIdx.x) & 3) * {k}) + ((((_tmp * 3) + "
        f"(((int)blockIdx.x) >> 4)) >> {row_tile_shift}) * 256)) + "
        "((tl::get_lane_idx(64) >> 3) * 32)) + (warp_idx * 4)) + "
        "((tl::get_lane_idx(64) & 7) >> 1))"
    )


def _prove_local_row_equivalence(rows: list[str], *, m: int, n: int, k: int) -> dict[str, int]:
    """Prove every accepted source row is tile-row + the target local row.

    The source anchor constrains the exact code-generation decomposition;
    exhaustive evaluation then covers every legal owner-interleaved job and
    all 512 hardware thread ids.  This is intentionally a proof precondition,
    not an algebraic assumption made after the rewrite.
    """
    expected = _source_row_anchor(k)
    if len(rows) != 12 or any(row != expected for row in rows):
        raise ValueError("A idxen rows do not all match the supported generated global-row anchor")
    tile_cols = n // TILE
    tile_count = (m // TILE) * tile_cols
    tiles_per_owner = tile_count // 4
    checked = 0
    for job in range(tile_count):
        block = job % COMPUTE_BLOCKS
        tmp = job // COMPUTE_BLOCKS
        tile = (job % 4) * tiles_per_owner + job // 4
        tile_row = tile // tile_cols
        global_row_prefix = ((block & 3) * k +
                             ((((tmp * 3) + (block >> 4)) >> (2 if k == 1024 else 1)) * TILE))
        for thread in range(512):
            lane = thread % 64
            warp = thread // 64
            local_row = ((lane >> 3) * 32) + (warp * 4) + ((lane & 7) >> 1)
            if global_row_prefix + local_row != tile_row * TILE + local_row:
                raise AssertionError(
                    f"row equivalence failed for job={job} thread={thread}: "
                    f"{global_row_prefix + local_row} != {tile_row * TILE + local_row}"
                )
            checked += 1
    return {"a_idxen_rows": len(rows), "legal_jobs": tile_count, "thread_ids_per_job": 512,
            "evaluated_pairs": checked}


def transform(source: str, *, m: int, n: int, k: int, local_row: bool = False) -> tuple[str, dict[str, object]]:
    """Perform the one permitted HIP-only change with structural assertions."""
    _validate_shape(m, n, k)
    tile_cols = n // TILE
    tile_count = (m // TILE) * tile_cols
    tiles_per_owner = tile_count // 4
    struct_bit = (k * 2).bit_length() - 1

    a_entry = f"  int32x4_t A_dcu_res = tl::make_wave_buffer_resource<{struct_bit}>(A + (0));\n"
    b_entry = "  int32x4_t B_dcu_res = tl::make_wave_buffer_resource(B + (0));\n"
    if source.count("A_dcu_res") != 13 or source.count("B_dcu_res") != 13:
        raise ValueError("expected one entry resource plus exactly 12 A and 12 B compute copies")
    source = _replace_once(source, a_entry, "", "entry A resource")
    source = _replace_once(source, b_entry, "", "entry B resource")

    loop_pattern = re.compile(
        r"(    for \(int _tmp = 0; _tmp < \(\([^\n]+\) / 48\); \+\+_tmp\) \{\n)"
    )
    matches = list(loop_pattern.finditer(source))
    if len(matches) != 1:
        raise ValueError(f"owner tile loop: expected exactly one _tmp compute loop, found {len(matches)}")
    declarations = f"""      const int audit_job = int(blockIdx.x) + _tmp * {COMPUTE_BLOCKS};
      const int audit_tile = (audit_job % 4) * {tiles_per_owner} + audit_job / 4;
      const int audit_tile_row = audit_tile / {tile_cols};
      const int audit_tile_col = audit_tile % {tile_cols};
      int32x4_t A_dcu_res = tl::make_wave_buffer_resource<{struct_bit}>(A + (audit_tile_row * {TILE} * {k}));
      int32x4_t B_dcu_res = tl::make_wave_buffer_resource(B + (audit_tile_col * {TILE}));
"""
    source = source[:matches[0].end()] + declarations + source[matches[0].end():]

    a_calls = _calls_with_resource(source, "tl::cp_async_gs_idxen<", "A_dcu_res", 4)
    b_calls = _calls_with_resource(source, "tl::cp_async_gs<", "B_dcu_res", 3)
    if len(a_calls) != 12:
        raise ValueError(f"expected 12 A idxen copies, found {len(a_calls)}")
    if len(b_calls) != 12:
        raise ValueError(f"expected 12 B flat copies, found {len(b_calls)}")

    source_rows = [arguments[3].strip() for _, _, _, arguments in a_calls]
    equivalence = (_prove_local_row_equivalence(source_rows, m=m, n=n, k=k)
                   if local_row else None)

    a_replacements: list[str] = []
    for _, _, call, arguments in a_calls:
        if "sizeof(half_t)" not in arguments[2]:
            raise ValueError("A idxen source offset is not a half-byte expression")
        old_row = arguments[3].strip()
        new_row = ("(((tl::get_lane_idx(64) >> 3) * 32) + (warp_idx * 4) + "
                   "((tl::get_lane_idx(64) & 7) >> 1))" if local_row else
                   f"({old_row} - audit_tile_row * {TILE})")
        replacement_args = [arguments[0], arguments[1], arguments[2],
                            f" {new_row}"]
        a_replacements.append(call[:call.index("(") + 1] + ",".join(replacement_args) + ")")
    source = _replace_calls(source, a_calls, a_replacements)

    b_calls = _calls_with_resource(source, "tl::cp_async_gs<", "B_dcu_res", 3)
    # Re-find after A replacements: positions are intentionally not reused.
    if len(b_calls) != 12:
        raise ValueError(f"expected 12 B flat copies after A rewrite, found {len(b_calls)}")
    b_replacements: list[str] = []
    for _, _, call, arguments in b_calls:
        old_byte_offset = arguments[2].strip()
        if "sizeof(half_t)" not in old_byte_offset:
            raise ValueError("B source offset is not a half-byte expression")
        replacement_args = [arguments[0], arguments[1],
                            f" (({old_byte_offset}) - audit_tile_col * {TILE} * 2)"]
        b_replacements.append(call[:call.index("(") + 1] + ",".join(replacement_args) + ")")
    source = _replace_calls(source, b_calls, b_replacements)

    if local_row:
        if source.count("audit_tile_row * 256") != 1:
            raise ValueError("local-row mode must retain tile row only in the A resource base")
        if source.count("((tl::get_lane_idx(64) >> 3) * 32) + (warp_idx * 4) + ((tl::get_lane_idx(64) & 7) >> 1)") != 12:
            raise ValueError("expected all 12 direct local-row A idxen expressions")
    elif source.count("audit_tile_row * 256") != 13:
        raise ValueError("expected A base plus all 12 A idxen row adjustments")
    if source.count("audit_tile_col * 256 * 2") != 12:
        raise ValueError("expected all 12 B byte-offset adjustments")
    if source.count("tl::ipc::opaque_thread_id") != 1:
        raise ValueError("communication opaque thread coordinate changed or was not unique")
    return source, {
        "a_idxen_copies": len(a_calls),
        "b_flat_copies": len(b_calls),
        "tile_count": tile_count,
        "tiles_per_owner": tiles_per_owner,
        "tile_cols": tile_cols,
        "struct_bit": struct_bit,
        "mode": "local-row" if local_row else "subtract",
        "local_row_equivalence": equivalence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="uppercase-A/B exported HIP source")
    parser.add_argument("--output", type=Path, required=True, help="tile-base HIP-only diagnostic output")
    parser.add_argument("--manifest", type=Path, help="defaults beside --output")
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--local-row", action="store_true",
                        help="replace A idxen rows with the proven direct native local-row expression")
    args = parser.parse_args()
    _validate_shape(args.m, args.n, args.k)
    source_path = args.source.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source HIP does not exist: {source_path}")
    if args.output.suffix != ".hip":
        raise ValueError("--output must end in .hip")
    if args.local_row and "local-row" not in args.output.name:
        raise ValueError("--local-row output filename must contain 'local-row' to preserve subtract evidence")

    original = source_path.read_text(encoding="utf-8")
    transformed, counts = transform(original, m=args.m, n=args.n, k=args.k,
                                    local_row=args.local_row)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(transformed, encoding="utf-8")
    manifest = (args.manifest.expanduser().resolve() if args.manifest else
                output.with_suffix(".manifest.json"))
    manifest.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema": "gemm-tile-base-hip-only-audit-v1",
        "diagnostic_only": True,
        "generated_by": str(Path(__file__).resolve()),
        "source": str(source_path),
        "source_sha256": _sha256_text(original),
        "output": str(output),
        "output_sha256": _sha256_text(transformed),
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "intervention": {
            "resource_construction": "entry -> owner tile loop",
            "a_idxen_row": ("direct native local-row after exhaustive equivalence proof"
                            if args.local_row else "subtract tile_row * 256"),
            "b_byte_offset": "subtract tile_col * 256 * 2",
            "communication_schedule_compute_barriers_unchanged": True,
        },
        "asserted_counts": counts,
    }
    manifest.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
