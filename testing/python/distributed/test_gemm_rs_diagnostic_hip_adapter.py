"""Host-only ABI regression test for the exported GEMM+RS HIP diagnostic."""
from pathlib import Path
import sys
import unittest


PROTOTYPE_DIR = Path(__file__).resolve().parents[3] / "examples/distributed/gemm_rs_prototype"
sys.path.insert(0, str(PROTOTYPE_DIR))
from gemm_rs_diagnostic_hip_abi import diagnostic_hip_arguments


class _Pointer:
    def __init__(self, value: int):
        self.value = value

    def data_ptr(self) -> int:
        return self.value


class _Allocation(_Pointer):
    def __init__(self, data: int, descriptor: int | None = None):
        super().__init__(data)
        self.descriptor = _Pointer(descriptor) if descriptor is not None else None

    def device_view(self):
        assert self.descriptor is not None
        return self.descriptor


class DiagnosticHipAbiTest(unittest.TestCase):
    def test_external_kernel_uses_descriptors_before_local_payloads(self):
        args = diagnostic_hip_arguments(
            _Allocation(101), _Allocation(102), _Allocation(103, 203),
            _Allocation(104, 204), _Allocation(105), 7, 4,
        )
        self.assertEqual(args, (101, 102, 203, 204, 105, 103, 104, 7, 4))


if __name__ == "__main__":
    unittest.main()
