"""Standalone CPU checks for ABI2 IPC metadata."""
import importlib.util
from pathlib import Path
import sys
import unittest

def load():
    name = "_ipc_contracts"
    if name in sys.modules:
        return sys.modules[name]
    directory = Path(__file__).resolve().parents[3] / "tilelang" / "distributed"
    spec = importlib.util.spec_from_file_location(name, directory / "__init__.py", submodule_search_locations=[str(directory)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

d = load()

class TestIPCContracts(unittest.TestCase):
    def test_reference_configs_and_independent_formulas(self):
        configs = (("ar_v2", (48,24,16,8)), ("rs_v2", (32,32,8,4)), ("ag_v2", (32,32,24,16)))
        for profile, comms in configs:
            for s, comm in zip((2048,4096,8192,16384), comms):
                spec = d.IPCFusionSpec.reference_shape(profile, s, comm_blocks=comm)
                plan = d.IPCSchedulePlan(spec)
                self.assertEqual(plan.compute_blocks, 80-comm)
                self.assertEqual(plan.flags_count, spec.flags_payload_count+4+1024)
                if profile == "ar_v2":
                    self.assertEqual(plan.owner(3), 3)
                    self.assertEqual(plan.ready_index(5,2), 22)
                elif profile == "rs_v2":
                    job = 5
                    self.assertEqual(plan.producer_tile(job), (job%4)*(plan.tile_count//4)+job//4)
                else:
                    transfer = plan.ag_transfer(2, 0)
                    self.assertEqual((transfer.row, transfer.source), (2*plan.ag_rows_per_rank, 2))
                    self.assertEqual(plan.ag_transfer_job(2, transfer.row, transfer.chunk), 0)
    def test_inverse_offsets_and_ag_fail_closed(self):
        plan = d.IPCSchedulePlan(d.IPCFusionSpec.reference_shape("rs_v2", 2048, comm_blocks=32))
        for tile in range(plan.tile_count):
            self.assertEqual(plan.coords_tile(*plan.tile_coords(tile)), tile)
            self.assertEqual(plan.tile_from_owned(plan.owner(tile), plan.owned_index(tile)), tile)
            self.assertEqual(plan.partial_offset_bytes(tile), tile*65536*2)
        ag = d.IPCSchedulePlan(d.IPCFusionSpec.reference_shape("ag_v2", 2048, comm_blocks=32))
        for method, args in ((ag.ready_index,(0,0)), (ag.partial_offset_bytes,(0,)), (ag.compute_job,(0,0)), (ag.owner,(0,))):
            with self.assertRaises(d.DistributedConfigError): method(*args)
    def test_manifest_and_bindings_fail_closed(self):
        spec = d.IPCFusionSpec.reference_shape("ar_v2", 2048, comm_blocks=48)
        abi = d.IPCKernelABI(spec, d.IPCSchedulePlan(spec))
        self.assertEqual(d.IPCKernelABI.from_json(abi.to_json()).manifest(), abi.manifest())
        raw = abi.to_dict(); raw["manifest"]["launch"]["threads"] = 1
        with self.assertRaises(d.DistributedConfigError): d.IPCKernelABI.from_dict(raw)
        bindings = [d.IPCDeviceBinding(x.role,x.dtype,x.shape,x.strides,x.layout,x.memory_kind,x.nbytes,0,x.alignment,x.ipc_allocation,i+1,2,"ctx",0) for i,x in enumerate(abi.requirements)]
        abi.validate_bindings(bindings, expected_generation=2, expected_context="ctx", expected_rank=0)
        with self.assertRaises(d.DistributedConfigError): d.IPCFusionSpec(profile="ag_v2", k=64)
        with self.assertRaises(d.DistributedConfigError): d.IPCFusionSpec(nranks=True)

if __name__ == "__main__":
    unittest.main()
