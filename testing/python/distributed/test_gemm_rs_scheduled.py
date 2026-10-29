"""CPU/source-structure checks for the bounded matched GEMM+RS candidate."""
import importlib.util
import inspect
from pathlib import Path
import sys


PROTOTYPE_DIR = Path(__file__).resolve().parents[3] / "examples" / "distributed" / "gemm_rs_prototype"


def _load_module(name: str):
    path = PROTOTYPE_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_matched_schedule_maps_jobs_to_all_tiles_and_continuous_row_owners():
    original = sys.modules.get("gemm_reduce_scatter")
    scheduled_original = sys.modules.get("gemm_reduce_scatter_scheduled")
    try:
        # The candidate imports the established constants, but this test only
        # exercises Python schedule formulas; no GPU/lowering is involved.
        base = _load_module("gemm_reduce_scatter")
        mod = _load_module("gemm_reduce_scatter_scheduled")
        params = mod.schedule_parameters(2048, 2048, 80, 32)
        assert params["tile_count"] == 64
        assert params["compute_blocks"] == 48
        assert params["tiles_per_rank"] == 16
        jobs = []
        for block in range(params["compute_blocks"]):
            jobs.extend(range(block, params["tile_count"], params["compute_blocks"]))
        assert sorted(jobs) == list(range(params["tile_count"]))
        tiles = [mod.interleaved_tile(job, params["tile_count"]) for job in jobs]
        assert sorted(tiles) == list(range(params["tile_count"]))
        assert [mod.owner_for_tile(mod.interleaved_tile(job, params["tile_count"]), params["tile_count"])
                for job in range(4)] == [0, 1, 2, 3]

        # Each owner gets an adjacent tile interval and converts it to its
        # local consecutive row range (not an owner-cyclic output mapping).
        tile_cols = 2048 // base.TILE_N
        for owner in range(base.PE_COUNT):
            owner_tiles = range(owner * params["tiles_per_rank"], (owner + 1) * params["tiles_per_rank"])
            assert all(mod.owner_for_tile(tile, params["tile_count"]) == owner for tile in owner_tiles)
            rows = [(tile // tile_cols) % params["tile_rows_per_rank"] for tile in owner_tiles]
            assert rows == [row for row in range(params["tile_rows_per_rank"]) for _ in range(tile_cols)]
    finally:
        for name, saved in (("gemm_reduce_scatter", original),
                            ("gemm_reduce_scatter_scheduled", scheduled_original)):
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved


def test_matched_flags_are_non_overlapping_and_protocol_keeps_both_roles():
    original = sys.modules.get("gemm_reduce_scatter")
    scheduled_original = sys.modules.get("gemm_reduce_scatter_scheduled")
    try:
        _load_module("gemm_reduce_scatter")
        mod = _load_module("gemm_reduce_scatter_scheduled")
        layout = mod.flag_layout(64, 32)
        assert layout == {
            "tile_ready_words": 256,
            "peer_done_base": 256,
            "block_done_base": 260,
            "flag_words": 292,
        }
        tile_ready = set(range(layout["tile_ready_words"]))
        peer_done = set(range(layout["peer_done_base"], layout["block_done_base"]))
        block_done = set(range(layout["block_done_base"], layout["flag_words"]))
        assert not tile_ready & peer_done
        assert not tile_ready & block_done
        assert not peer_done & block_done

        source = inspect.getsource(mod.gemm_reduce_scatter_scheduled)
        assert "if block_idx < compute_blocks:" in source
        assert "else:" in source
        assert "for job in T.serial(block_idx, tile_count, compute_blocks):" in source
        assert "num_stages=4" in source
        assert 'T.call_extern("int32", "tl::ipc::opaque_thread_id", tx)' in source
        assert "for linear in T.serial(comm_tx, TILE_M * TILE_N, THREADS):" in source
        assert "T.signal_set(ready[flags[\"block_done_base\"] + lane], 1)" in source
        assert "T.wait_eq(ready[flags[\"peer_done_base\"] + 3], 1)" in source
    finally:
        for name, saved in (("gemm_reduce_scatter", original),
                            ("gemm_reduce_scatter_scheduled", scheduled_original)):
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved


def test_comm_ctas_cover_owner_tiles_with_idle_lanes_and_multiple_rounds():
    original = sys.modules.get("gemm_reduce_scatter")
    scheduled_original = sys.modules.get("gemm_reduce_scatter_scheduled")
    try:
        _load_module("gemm_reduce_scatter")
        mod = _load_module("gemm_reduce_scatter_scheduled")
        # 2048 has 16 tiles per owner but 32 communication lanes: half the
        # CTAs have no tile work and must still reach block_done/peer_done.
        small = mod.schedule_parameters(2048, 2048, 80, 32)
        small_work = [list(range(lane, small["tiles_per_rank"], small["comm_blocks"]))
                      for lane in range(small["comm_blocks"])]
        assert sorted(tile for lane_work in small_work for tile in lane_work) == list(range(16))
        assert all(not lane_work for lane_work in small_work[16:])

        # 4096 gives every lane two distinct rounds.  Per owner the exact
        # sequence remains local 0..63, so rank*tiles_per_rank produces the
        # continuous row-shard tiles without overlap.
        large = mod.schedule_parameters(4096, 4096, 80, 32)
        assert large["tiles_per_rank"] == 64
        rounds = [list(range(lane, large["tiles_per_rank"], large["comm_blocks"]))
                  for lane in range(large["comm_blocks"])]
        assert all(len(lane_work) == 2 for lane_work in rounds)
        assert sorted(tile for lane_work in rounds for tile in lane_work) == list(range(64))
        for rank in range(4):
            owned = [rank * large["tiles_per_rank"] + tile
                     for lane_work in rounds for tile in lane_work]
            assert sorted(owned) == list(range(rank * 64, (rank + 1) * 64))
    finally:
        for name, saved in (("gemm_reduce_scatter", original),
                            ("gemm_reduce_scatter_scheduled", scheduled_original)):
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved
