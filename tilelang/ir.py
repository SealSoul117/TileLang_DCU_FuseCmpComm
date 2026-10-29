from tilelang import tvm as tvm
from tvm.ir.base import Node
from tvm.runtime import Scriptable
from tilelang._ffi_compat import register_object
from tvm.target import Target
from tilelang import _ffi_api


@register_object("tl.Fill")
class Fill(Node, Scriptable): ...


@register_object("tl.AtomicAdd")
class AtomicAdd(Node, Scriptable): ...


@register_object("tl.Copy")
class Copy(Node, Scriptable): ...


@register_object("tl.Conv2DIm2Col")
class Conv2DIm2ColOp(Node, Scriptable): ...


@register_object("tl.GemmWarpPolicy")
class GemmWarpPolicy(Node, Scriptable):
    policy_type: int
    m_warp: int
    n_warp: int

    def compute_warp_partition(self, M: int, N: int, block_size: int, target: Target, is_wgmma: bool):
        _ffi_api.GemmWarpPolicyComputeWarpPartition(self, int(M), int(N), int(block_size), target, is_wgmma)
        return self.m_warp, self.n_warp


@register_object("tl.GemmSPWarpPolicy")
class GemmSPWarpPolicy(Node, Scriptable):
    policy_type: int
    m_warp: int
    n_warp: int

    def compute_warp_partition(self, M: int, N: int, block_size: int, target: Target, is_wgmma: bool, bits: int):
        _ffi_api.GemmSPWarpPolicyComputeWarpPartition(self, int(M), int(N), int(block_size), target, is_wgmma, bits)
        return self.m_warp, self.n_warp


@register_object("tl.Gemm")
class Gemm(Node, Scriptable): ...


@register_object("tl.GemmSP")
class GemmSP(Node, Scriptable): ...


@register_object("tl.FinalizeReducerOp")
class FinalizeReducerOp(Node, Scriptable): ...


@register_object("tl.ParallelOp")
class ParallelOp(Node, Scriptable): ...


@register_object("tl.ReduceOp")
class ReduceOp(Node, Scriptable): ...


@register_object("tl.CumSumOp")
class CumSumOp(Node, Scriptable): ...


@register_object("tl.RegionOp")
class RegionOp(Node, Scriptable): ...


@register_object("tl.ReduceType")
class ReduceType(Node, Scriptable): ...
