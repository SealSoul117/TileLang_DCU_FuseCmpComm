"""CPU-only M0 contract tests.

The loader imports ``tilelang/distributed`` as a standalone package, so these
tests do not execute the GPU-/DTK-dependent ``tilelang/__init__.py``.
Run with either:

    python3 -m unittest testing/python/distributed/test_m0_contracts.py
    python3 -m pytest testing/python/distributed/test_m0_contracts.py
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import importlib.util
import json
from pathlib import Path
import sys
import unittest


def _load_contract_package():
    name = "_tilelang_m0_contracts"
    if name in sys.modules:
        return sys.modules[name]
    root = Path(__file__).resolve().parents[3]
    package_dir = root / "tilelang" / "distributed"
    module_spec = importlib.util.spec_from_file_location(
        name, package_dir / "__init__.py", submodule_search_locations=[str(package_dir)]
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module


d = _load_contract_package()


class SchedulePlanTest(unittest.TestCase):
    def test_default_v1_mapping_matches_reference_formulas(self):
        spec = d.FusionSpec()
        plan = d.SchedulePlan(spec, compute_blocks=61)  # 80 total blocks - 19 comm blocks
        self.assertEqual((plan.tiles_x, plan.tiles_y, plan.tile_count), (32, 32, 1024))
        producer = plan.producer_for_tile(1023, source=3)
        self.assertEqual((producer.bx, producer.by), (31, 31))
        self.assertEqual((producer.owner, producer.owned_index, producer.owner_slot), (3, 255, 1023))
        self.assertEqual((producer.producer_block, producer.producer_iteration, producer.batch_start_work), (47, 16, 1023))
        ready = plan.ready_for_tile(1023, source=3)
        self.assertEqual((ready.owner, ready.batch_start_work, ready.source, ready.storage_index), (3, 1023, 3, 4095))

    def test_coords_to_tile_is_validated_inverse_across_grids(self):
        specs = (
            d.FusionSpec(m=384, n=640, tile_shape=(128, 128), nranks=3, comm_blocks=1),
            d.FusionSpec(m=256, n=384, tile_shape=(128, 128), nranks=4, comm_blocks=2),
        )
        for spec in specs:
            plan = d.SchedulePlan(spec, compute_blocks=3)
            for tile_id in range(plan.tile_count):
                bx, by = plan.tile_to_coords(tile_id)
                self.assertEqual(plan.coords_to_tile(bx, by), tile_id)
            for bx, by in ((-1, 0), (plan.tiles_x, 0), (0, -1), (0, plan.tiles_y), (True, 0), (0, None)):
                with self.assertRaises(d.DistributedConfigError):
                    plan.coords_to_tile(bx, by)
        plan = d.SchedulePlan(d.FusionSpec(m=128, n=128, tile_shape=(128, 128), comm_blocks=1), 1)
        with self.assertRaises(d.DistributedConfigError):
            plan.ready_key(None, 0, 0)

    def test_batch_formula_and_owner_deduplicated_notifications(self):
        spec = d.FusionSpec(m=384, n=512, nranks=2, tile_shape=(128, 128), comm_blocks=1, batch_tiles=4)
        plan = d.SchedulePlan(spec, compute_blocks=2)
        # Reference formula: block + floor(iteration / batch) * batch * compute_blocks.
        assignment = plan.producer_for_work(6, source=1)
        self.assertEqual((assignment.producer_block, assignment.producer_iteration, assignment.batch_start_work), (0, 3, 0))
        self.assertEqual(plan.batch_work_items(0, 0), (0, 2, 4, 6))
        # 0, 2, 4, 6 all have owner 0 under cyclic P=2: one notification, not four.
        notifications = plan.notifications_for_batch(0, 0, source=1)
        self.assertEqual(len(notifications), 1)
        self.assertEqual((notifications[0].owner, notifications[0].storage_index), (0, 1))
        # The tail batch has only existing work items and remains correctly aligned.
        self.assertEqual(plan.batch_work_items(1, 9), (9, 11))

    def test_enumerated_mapping_bijections_and_ready_coverage(self):
        for nranks in range(1, 6):
            for compute_blocks in range(1, 5):
                for batch_tiles in (1, 2, 4, 8):
                    spec = d.FusionSpec(
                        m=384,
                        n=640,
                        nranks=nranks,
                        tile_shape=(128, 128),
                        comm_blocks=1,
                        batch_tiles=batch_tiles,
                    )
                    plan = d.SchedulePlan(spec, compute_blocks)
                    self.assertEqual([plan.work_to_tile(work) for work in range(plan.tile_count)], list(range(plan.tile_count)))
                    slots = set()
                    for tile in range(plan.tile_count):
                        owner = plan.owner_of_tile(tile)
                        self.assertEqual(plan.tile_from_owned_index(owner, plan.owned_index(tile)), tile)
                        for source in range(nranks):
                            producer = plan.producer_for_tile(tile, source)
                            self.assertIn(producer.work_id, plan.batch_work_items(producer.producer_block, producer.batch_start_work))
                            ready = plan.ready_for_tile(tile, source)
                            self.assertEqual(ready.owner, owner)
                            self.assertEqual(ready.storage_index, ready.batch_start_work * nranks + source)
                            slots.add((owner, producer.owner_slot))
                    self.assertEqual(len(slots), plan.tile_count * nranks)
                    consumers = tuple(plan.consumer_tile_order())
                    self.assertEqual({tile for _, _, tile in consumers}, set(range(plan.tile_count)))
                    self.assertEqual(len(consumers), plan.tile_count)

    def test_reduce_scatter_uses_contiguous_rows_and_separate_interleaving(self):
        spec = d.FusionSpec(
            m=512,
            n=384,
            nranks=3,
            tile_shape=(128, 128),
            collective_kind="reduce_scatter",
            owner_policy="contiguous_rows",
            mathematical_output_distribution="row_sharded",
            compute_tile_order="owner_interleaved",
            comm_blocks=2,
            batch_tiles=2,
        )
        plan = d.SchedulePlan(spec, compute_blocks=5)
        # Two row tiles belong to rank 0, then one row each to ranks 1 and 2.
        self.assertEqual([plan.owner_of_tile(row * plan.tiles_x) for row in range(plan.tiles_y)], [0, 0, 1, 2])
        order = plan.consumer_tile_order()
        self.assertEqual(order[:3], ((0, 0, 0), (1, 0, 6), (2, 0, 9)))
        self.assertEqual(
            order[3:],
            ((0, 1, 1), (1, 1, 7), (2, 1, 10), (0, 2, 2), (1, 2, 8), (2, 2, 11), (0, 3, 3), (0, 4, 4), (0, 5, 5)),
        )
        # A ready key retains a producer work batch start.  This unified M0
        # protocol is not asserted to be byte-for-byte compatible with any
        # handwritten RS ready layout.
        producer = plan.producer_for_tile(6, source=2)
        self.assertEqual(producer.batch_start_work, 1)
        self.assertEqual(plan.ready_for_tile(6, source=2).storage_index, 1 * 3 + 2)

    def test_edge_mapping_and_inactive_owned_slots_are_explicit(self):
        edge_spec = d.FusionSpec(m=130, n=257, nranks=4, tile_shape=(128, 128), comm_blocks=1)
        edge_plan = d.SchedulePlan(edge_spec, compute_blocks=2)
        self.assertTrue(edge_spec.requires_edge_guards)
        self.assertEqual(edge_plan.tile_extent(5), (2, 1))
        with self.assertRaisesRegex(d.DistributedConfigError, "does not yet implement"):
            edge_spec.validate_v1_lowering()

        rs_spec = d.FusionSpec(
            m=128,
            n=256,
            nranks=4,
            tile_shape=(128, 128),
            collective_kind="reduce_scatter",
            owner_policy="contiguous_rows",
            mathematical_output_distribution="row_sharded",
            compute_tile_order="owner_interleaved",
            comm_blocks=1,
        )
        rs_plan = d.SchedulePlan(rs_spec, compute_blocks=1)
        self.assertEqual(rs_plan.owner_tiles(2), ())
        self.assertIsNone(rs_plan.tile_from_owned_index(2, 0))
        self.assertEqual(tuple(rs_plan.consumer_for_lane(2, 0)), ())

        # More compute blocks than tiles and more comm lanes than owned tiles
        # are legal mapping tails; they produce no fictitious assignments.
        sparse = d.SchedulePlan(d.FusionSpec(m=128, n=256, nranks=4, tile_shape=(128, 128), comm_blocks=3), 8)
        self.assertEqual(tuple(sparse.producer_batches(7)), ())
        self.assertEqual(tuple(sparse.consumer_for_lane(0, 2)), ())

    def test_rs_compute_order_is_owner_interleaved_and_bijective(self):
        spec = d.FusionSpec(
            m=512,
            n=384,
            nranks=4,
            tile_shape=(128, 128),
            collective_kind="reduce_scatter",
            owner_policy="contiguous_rows",
            mathematical_output_distribution="row_sharded",
            compute_tile_order="owner_interleaved",
            comm_blocks=2,
            batch_tiles=2,
        )
        plan = d.SchedulePlan(spec, compute_blocks=5)
        self.assertEqual([plan.owner_of_tile(plan.work_to_tile(work)) for work in range(4)], [0, 1, 2, 3])
        self.assertEqual([plan.work_to_tile(work) for work in range(4)], [0, 3, 6, 9])
        self.assertEqual([plan.tile_to_work(tile) for tile in range(plan.tile_count)], [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11])
        self.assertEqual([plan.tile_to_coords(plan.work_to_tile(work)) for work in range(4)], [(0, 0), (0, 1), (0, 2), (0, 3)])

    def test_ready_notifications_cover_exact_batches_for_both_orders(self):
        specs = (
            d.FusionSpec(m=384, n=640, nranks=3, tile_shape=(128, 128), comm_blocks=2, batch_tiles=4),
            d.FusionSpec(
                m=640,
                n=384,
                nranks=4,
                tile_shape=(128, 128),
                collective_kind="reduce_scatter",
                owner_policy="contiguous_rows",
                mathematical_output_distribution="row_sharded",
                compute_tile_order="owner_interleaved",
                comm_blocks=2,
                batch_tiles=4,
            ),
        )
        for spec in specs:
            for compute_blocks in (1, 3, 4):
                plan = d.SchedulePlan(spec, compute_blocks)
                emitted = {}
                emission_count = {}
                for source in range(spec.nranks):
                    for block in range(compute_blocks):
                        for active_work in plan.producer_batches(block):
                            batch_start = active_work[0]
                            notifications = plan.notifications_for_batch(block, batch_start, source)
                            self.assertEqual(len(notifications), len(set(notifications)))
                            for key in notifications:
                                emission_count[key] = emission_count.get(key, 0) + 1
                                emitted[key] = {
                                    plan.work_to_tile(work)
                                    for work in active_work
                                    if plan.owner_of_tile(plan.work_to_tile(work)) == key.owner
                                }
                self.assertTrue(all(count == 1 for count in emission_count.values()))
                for tile in range(plan.tile_count):
                    for source in range(spec.nranks):
                        key = plan.ready_for_tile(tile, source)
                        self.assertIn(key, emitted)
                        self.assertIn(tile, emitted[key])
                        producer = plan.producer_for_tile(tile, source)
                        self.assertEqual(key.batch_start_work, producer.batch_start_work)


class ContractTest(unittest.TestCase):
    def test_buffer_units_nonzero_offset_and_span(self):
        desc = d.SymmetricBufferDesc(
            allocation_id="stage-0",
            dtype="fp32",
            shape=(2, 3),
            strides=(5, 1),
            alignment=256,
            byte_offset=16,
            nbytes=48,
            runtime_generation=7,
        )
        self.assertEqual(desc.stride_unit, "elements")
        self.assertEqual(desc.view_span_nbytes, 32)
        self.assertEqual(desc.byte_offset + desc.view_span_nbytes, desc.nbytes)
        self.assertEqual(d.SymmetricBufferDesc.from_json(desc.to_json()), desc)
        with self.assertRaisesRegex(d.DistributedConfigError, "exceeds allocation"):
            d.SymmetricBufferDesc("stage-0", "fp32", (2, 3), (5, 1), 256, 16, 47)
        with self.assertRaisesRegex(d.DistributedConfigError, "positive"):
            d.SymmetricBufferDesc("stage-0", "fp32", (2, 3), (0, 1), 256, 0, 32)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaisesRegex(d.DistributedConfigError, "allgather"):
            d.FusionSpec(collective_kind="allgather")
        with self.assertRaisesRegex(d.DistributedConfigError, "batch_tiles"):
            d.FusionSpec(batch_tiles=3)
        with self.assertRaisesRegex(d.DistributedConfigError, "owner_policy"):
            d.FusionSpec(collective_kind="reduce_scatter")
        with self.assertRaisesRegex(d.DistributedConfigError, "mathematical_output_distribution"):
            d.FusionSpec(
                collective_kind="reduce_scatter",
                owner_policy="contiguous_rows",
                compute_tile_order="owner_interleaved",
            )

    def test_abi_json_immutability_identity_and_runtime_validation(self):
        spec = d.FusionSpec()
        self.assertEqual(d.FusionSpec.from_json(spec.to_json()), spec)
        plan = d.SchedulePlan(spec, compute_blocks=61)
        self.assertEqual(d.SchedulePlan.from_json(plan.to_json()), plan)
        abi = d.DistributedKernelABI.v1(spec, plan)
        self.assertEqual(d.DistributedKernelABI.from_json(abi.to_json()), abi)
        with self.assertRaises(FrozenInstanceError):
            abi.launch = d.LaunchConfig(80)  # type: ignore[misc]

        renamed = d.DistributedKernelABI.v1(
            spec,
            plan,
            allocation_ids={"staging": "rank-local-stage", "output": "rank-local-output", "ready": "rank-local-ready"},
            runtime_generation=9,
        )
        self.assertEqual(abi.compile_identity(), renamed.compile_identity())
        self.assertNotEqual(abi.agreement_identity(), renamed.agreement_identity())
        runtime = d.RuntimeBinding(
            rank=0,
            nranks=4,
            context_handle="context-0",
            allocation_pointers=(("staging", "0x100"), ("output", "0x200"), ("ready", "0x300")),
            allocation_generations=(("staging", 0), ("output", 0), ("ready", 0)),
        )
        self.assertIsNone(abi.bind_runtime(runtime))
        with self.assertRaisesRegex(d.DistributedConfigError, "generation"):
            abi.bind_runtime(replace(runtime, allocation_generations=(("staging", 1), ("output", 0), ("ready", 0))))
        with self.assertRaisesRegex(d.DistributedConfigError, "grid_blocks"):
            replace(abi, launch=d.LaunchConfig(79, 256, 65536))

    def test_v1_abi_explicitly_rejects_unimplemented_edge_lowering(self):
        spec = d.FusionSpec(m=129, n=128, tile_shape=(128, 128), comm_blocks=1)
        plan = d.SchedulePlan(spec, compute_blocks=2)
        with self.assertRaisesRegex(d.DistributedConfigError, "does not yet implement"):
            d.DistributedKernelABI.v1(spec, plan, grid_blocks=3)

    def test_rs_output_local_offsets_and_schedule_derived_workspace(self):
        spec = d.FusionSpec(
            m=512,
            n=384,
            nranks=3,
            tile_shape=(128, 128),
            collective_kind="reduce_scatter",
            owner_policy="contiguous_rows",
            mathematical_output_distribution="row_sharded",
            compute_tile_order="owner_interleaved",
            comm_blocks=2,
        )
        plan = d.SchedulePlan(spec, compute_blocks=5)
        abi = d.DistributedKernelABI.v1(spec, plan, grid_blocks=7)
        output = next(item for item in abi.workspace if item.role == "output")
        self.assertEqual(output.shape, (6, 128, 128))  # symmetric max owner-local capacity
        self.assertEqual([plan.output_valid_tile_count(rank) for rank in range(3)], [6, 3, 3])
        self.assertEqual(plan.output_tile_to_local_offset(0, 0), 0)
        self.assertEqual(plan.output_tile_to_local_offset(6, 1), 0)
        self.assertEqual(plan.output_tile_to_local_offset(9, 2), 0)
        self.assertIsNone(plan.output_tile_to_local_offset(0, 1))

        shrunk_ready = d.WorkspaceRequirement("ready", "uint32", (1,), (1,))
        shrunk_binding = d.BufferBinding(
            "ready",
            "ready",
            d.SymmetricBufferDesc("ready", "uint32", (1,), (1,), 256, 0, 4),
        )
        workspace = tuple(shrunk_ready if item.role == "ready" else item for item in abi.workspace)
        bindings = tuple(shrunk_binding if item.role == "ready" else item for item in abi.bindings)
        with self.assertRaisesRegex(d.DistributedConfigError, "derived from spec and schedule"):
            replace(abi, workspace=workspace, bindings=bindings)

    def test_abi_rejects_alias_capability_permission_alignment_and_tile_mismatch(self):
        spec = d.FusionSpec()
        plan = d.SchedulePlan(spec, compute_blocks=61)
        abi = d.DistributedKernelABI.v1(spec, plan)
        with self.assertRaisesRegex(d.DistributedConfigError, "distinct"):
            d.DistributedKernelABI.v1(spec, plan, allocation_ids={"staging": "same", "output": "same", "ready": "same"})
        with self.assertRaisesRegex(d.DistributedConfigError, "requires capabilities"):
            replace(abi, required_capabilities=("anything",))
        staging = abi.bindings[0]
        readonly = replace(staging, buffer=replace(staging.buffer, permissions=("read",)))
        with self.assertRaisesRegex(d.DistributedConfigError, "permit both"):
            replace(abi, bindings=(readonly,) + abi.bindings[1:])
        misaligned = replace(staging, buffer=replace(staging.buffer, alignment=8))
        with self.assertRaisesRegex(d.DistributedConfigError, "effective view alignment"):
            replace(abi, bindings=(misaligned,) + abi.bindings[1:])
        odd_tile = d.FusionSpec(m=256, n=128, k_local=1024, tile_shape=(256, 128), comm_blocks=1)
        with self.assertRaisesRegex(d.DistributedConfigError, "tile_shape"):
            d.DistributedKernelABI.v1(odd_tile, d.SchedulePlan(odd_tile, 2), grid_blocks=3)

    def test_strict_json_schema_and_boolean_rejections(self):
        spec = d.FusionSpec()
        plan = d.SchedulePlan(spec, 61)
        abi = d.DistributedKernelABI.v1(spec, plan)
        for payload in ("{", "[]", "{}"):
            with self.assertRaises(d.DistributedConfigError):
                d.DistributedKernelABI.from_json(payload)
        unknown = abi.to_dict()
        unknown["unknown"] = 1
        with self.assertRaisesRegex(d.DistributedConfigError, "unknown fields"):
            d.DistributedKernelABI.from_dict(unknown)
        forged = abi.to_dict()
        forged["workspace"][0]["min_nbytes"] += 4
        with self.assertRaisesRegex(d.DistributedConfigError, "min_nbytes"):
            d.DistributedKernelABI.from_dict(forged)
        with self.assertRaises(d.DistributedConfigError):
            d.FusionSpec(m=True)
        with self.assertRaises(d.DistributedConfigError):
            d.SymmetricBufferDesc("x", "fp32", (1,), (1,), 4, 0, 4, symmetric="false")
        with self.assertRaises(d.DistributedConfigError):
            replace(abi, abi_version=True)
        with self.assertRaises(d.DistributedConfigError):
            d.RuntimeBinding(True, 4, "ctx", (("staging", "0x1"),), (("staging", 0),))
        malformed_spec = spec.to_dict()
        malformed_spec["collective_kind"] = []
        with self.assertRaises(d.DistributedConfigError):
            d.FusionSpec.from_json(json.dumps(malformed_spec))
        malformed_desc = d.SymmetricBufferDesc("x", "fp32", (1,), (1,), 4, 0, 4).to_dict()
        malformed_desc["permissions"] = [[]]
        with self.assertRaises(d.DistributedConfigError):
            d.SymmetricBufferDesc.from_json(json.dumps(malformed_desc))


if __name__ == "__main__":
    unittest.main()
