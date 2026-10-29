"""Parent review: independent formulas from the three handwritten IPC kernels."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import unittest


def load_contracts():
    name = "_tilelang_ipc_parent_review"
    if name in sys.modules:
        return sys.modules[name]
    directory = Path(__file__).resolve().parents[3] / "tilelang/distributed"
    spec = importlib.util.spec_from_file_location(name, directory / "__init__.py",
                                                submodule_search_locations=[str(directory)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


d = load_contracts()


class IPCIndependentReview(unittest.TestCase):
    def test_ar_rs_against_reference_formulas_all_eight_shapes(self):
        for profile, comms in (("ar_v2", (48, 24, 16, 8)), ("rs_v2", (32, 32, 8, 4))):
            for size, comm in zip((2048, 4096, 8192, 16384), comms):
                spec = d.IPCFusionSpec.reference_shape(profile, size, comm_blocks=comm)
                plan = d.IPCSchedulePlan(spec)
                count = (size // 256) ** 2
                self.assertEqual(spec.global_shape, (size, size, size))
                self.assertEqual(plan.flags_count, count * 4 + 1028)
                for job in range(count):
                    tile = job if profile == "ar_v2" else (job % 4) * (count // 4) + job // 4
                    owner = tile % 4 if profile == "ar_v2" else tile // (count // 4)
                    self.assertEqual(plan.producer_tile(job), tile)
                    self.assertEqual(plan.producer_job(tile), job)
                    self.assertEqual(plan.owner(tile), owner)
                    self.assertEqual(plan.partial_offset_bytes(tile), tile * 131072)
                    for source in range(4):
                        self.assertEqual(plan.ready_index(tile, source), tile * 4 + source)
                    row, col = divmod(tile, size // 256)
                    if profile == "rs_v2":
                        row -= owner * (size // 1024)
                    view = plan.output_tile_view(tile)
                    self.assertEqual(view.offset_bytes, (row * 256 * size + col * 256) * 2)
                    self.assertEqual(view.row_stride_bytes, size * 2)
                    # The last element of a tile is strided, not origin+tile_bytes.
                    last = view.offset_bytes + 255 * view.row_stride_bytes + 255 * 2 + 2
                    output_rows = size // 4 if profile == "rs_v2" else size
                    self.assertLessEqual(last, output_rows * size * 2)

    def test_ag_transfers_queue_and_bounds_all_four_shapes(self):
        for size, comm in zip((2048, 4096, 8192, 16384), (32, 32, 24, 16)):
            spec = d.IPCFusionSpec.reference_shape("ag_v2", size, comm_blocks=comm)
            plan = d.IPCSchedulePlan(spec)
            rows, local_rows, chunks, cols = size // 256, size // 1024, size // 256, size // 1024
            self.assertEqual(spec.global_shape, (size, size, size))
            for rank in range(4):
                coverage = set()
                for job in range(rows * chunks):
                    row = (job // chunks + rank * local_rows) % rows
                    chunk = job % chunks
                    transfer = plan.ag_transfer(rank, job)
                    self.assertEqual((transfer.row, transfer.chunk, transfer.source), (row, chunk, row // local_rows))
                    source_offset = ((row % local_rows) * 256 * size + chunk * 65536) * 2
                    target_offset = (row * 256 * size + chunk * 65536) * 2
                    self.assertEqual(transfer.src_offset_bytes, source_offset)
                    self.assertEqual(transfer.dst_offset_bytes, target_offset)
                    self.assertLessEqual(source_offset + 131072, size // 4 * size * 2)
                    self.assertLessEqual(target_offset + 131072, size * size * 2)
                    self.assertEqual((transfer.counter_index, transfer.threshold), (row, chunks))
                    self.assertEqual(plan.ag_transfer_job(rank, row, chunk), job)
                    coverage.add((row, chunk))
                self.assertEqual(len(coverage), rows * chunks)
                tiles = set()
                for claim in range(rows * cols):
                    row, col = (claim // cols + rank * local_rows) % rows, claim % cols
                    assignment = plan.ag_queue_claim(rank, claim)
                    self.assertEqual((assignment.row, assignment.col), (row, col))
                    self.assertEqual(plan.ag_queue_claim_index(rank, row, col), claim)
                    tiles.add((row, col))
                self.assertEqual(len(tiles), rows * cols)

    def test_completion_slots_disjoint_at_max_grid(self):
        for profile in ("ar_v2", "rs_v2", "ag_v2"):
            spec = d.IPCFusionSpec(profile=profile, m=1024, n=256, k=256,
                                   cu_count=1024, comm_blocks=1023)
            plan = d.IPCSchedulePlan(spec)
            pe = {plan.pe_done_index(rank) for rank in range(4)}
            comm = {plan.comm_done_index(lane) for lane in range(1023)}
            self.assertFalse(pe & comm)
            self.assertLess(max(pe | comm), plan.flags_count)
            if profile == "ag_v2":
                self.assertNotIn(plan.ag_queue_index, pe | comm)
                self.assertEqual(plan.ag_queue_index, plan.flags_count - 1)

    def test_ag_rejects_ar_rs_only_ready_partial_and_static_assignment(self):
        plan = d.IPCSchedulePlan(d.IPCFusionSpec(profile="ag_v2"))
        for method, args in (("ready_index", (0, 0)), ("partial_offset_bytes", (0,)),
                             ("compute_job", (0, 0)), ("comm_owned", (0, 0)),
                             ("owner", (0,)), ("producer_tile", (0,))):
            with self.subTest(method=method), self.assertRaises(d.DistributedConfigError):
                getattr(plan, method)(*args)

    def test_fail_closed_profile_shapes_and_integer_types(self):
        invalid = ({"profile": []}, {"m": True}, {"n": 0}, {"k": -64},
                   {"nranks": 2}, {"batch_tiles": 2}, {"comm_blocks": 80},
                   {"cu_count": 1025}, {"m": 256, "n": 256},
                   {"profile": "ag_v2", "k": 320}, {"k": 1 << 40})
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(d.DistributedConfigError):
                d.IPCFusionSpec(**changes)

    def test_manifest_roundtrip_and_legacy_remains_version_one(self):
        self.assertEqual(d.ABI_VERSION, 1)
        self.assertEqual(d.IPC_ABI_VERSION, 2)
        for profile in ("ar_v2", "rs_v2", "ag_v2"):
            spec = d.IPCFusionSpec(profile=profile)
            abi = d.IPCKernelABI(spec, d.IPCSchedulePlan(spec))
            restored = d.IPCKernelABI.from_json(abi.to_json())
            self.assertEqual(restored.to_dict(), abi.to_dict())
            self.assertEqual(restored.compile_identity, abi.compile_identity)
            modified = replace(spec, comm_blocks=spec.comm_blocks + 1)
            other = d.IPCKernelABI(modified, d.IPCSchedulePlan(modified))
            self.assertNotEqual(other.compile_identity, abi.compile_identity)
            document = abi.to_dict()
            document["peer_va"] = 4096
            with self.assertRaises(d.DistributedConfigError):
                d.IPCKernelABI.from_dict(document)

    def test_manifest_rejects_equal_but_wrong_json_types(self):
        spec = d.IPCFusionSpec()
        abi = d.IPCKernelABI(spec, d.IPCSchedulePlan(spec))
        for field, invalid in (("full_resident", 1), ("threads", 512.0)):
            document = abi.to_dict()
            document["manifest"]["launch"][field] = invalid
            with self.subTest(field=field), self.assertRaises(d.DistributedConfigError):
                d.IPCKernelABI.from_dict(document)

    def test_json_rejects_duplicate_abi_version(self):
        spec = d.IPCFusionSpec()
        abi = d.IPCKernelABI(spec, d.IPCSchedulePlan(spec))
        document = '{"abi_version": 99,' + abi.to_json()[1:]
        with self.assertRaises(d.DistributedConfigError):
            d.IPCKernelABI.from_json(document)

    def _bindings(self):
        spec = d.IPCFusionSpec()
        abi = d.IPCKernelABI(spec, d.IPCSchedulePlan(spec))
        bindings = [d.IPCDeviceBinding(
            role=req.role, dtype=req.dtype, shape=req.shape, strides=req.strides,
            layout=req.layout, memory_kind=req.memory_kind, capacity_bytes=req.nbytes,
            offset_bytes=0, alignment=req.alignment, ipc_allocation=req.ipc_allocation,
            allocation_id=index + 1, generation=1, context_id="review", rank=0,
        ) for index, req in enumerate(abi.requirements)]
        return abi, bindings

    def test_bindings_reject_unrepresentable_native_fields(self):
        abi, bindings = self._bindings()
        index = next(i for i, binding in enumerate(bindings) if binding.role == "flags")
        for changes in ({"allocation_id": 1 << 80},
                        {"offset_bytes": 1 << 80,
                         "capacity_bytes": (1 << 80) + bindings[index].capacity_bytes}):
            with self.subTest(changes=changes), self.assertRaises(d.DistributedConfigError):
                modified = list(bindings)
                modified[index] = replace(bindings[index], **changes)
                abi.validate_bindings(modified, expected_rank=0, expected_context="review", expected_generation=1)

    def test_bindings_reject_bool_rank_expectation(self):
        abi, bindings = self._bindings()
        with self.assertRaises(d.DistributedConfigError):
            abi.validate_bindings(bindings, expected_rank=False, expected_context="review", expected_generation=1)


if __name__ == "__main__":
    unittest.main()
