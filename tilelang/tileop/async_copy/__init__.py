import tvm
from tvm import tir, target
from .node import Node, Scriptable
from . import _ffi_api
from tilelang._ffi_compat import register_object

@register_object("tl.AsyncCopyPy")
class AsyncCopyPy(Node, Scriptable):
    """
    AsyncCopyPy represents an asynchronous copy operation (e.g., Global to Shared).
    It follows the pattern of GemmPy for layout inference and lowering.
    """

    # 映射 FFI 字段（假设 C++ 侧定义的字段名为 src, dst, vectorSize 等）
    @property
    def source(self):
        return self.src

    @property
    def destination(self):
        return self.dst

    @property
    def vector_size(self):
        return self.vectorSize

    def infer_layout(self, target: tvm.target.Target, thread_nums: int):
        """推导布局：决定每个线程搬运哪些数据"""
        print(f"AsyncCopyPy infer_layout Target: {target}, thread_nums: {thread_nums}")
        
        # 1. 选择指令类型 (例如 cp.async, ldgsts, 或者自定义的 swizzled_copy)
        copy_inst = self._select_copy_instruction(target, thread_nums)
        
        # 2. 获取具体的实现类
        impl_class = self._get_implementation_class(copy_inst, target)
        
        # 3. 执行布局推理
        return impl_class(self).infer_layout(target, thread_nums)

    def lower(self, layout_map: dict, target: tvm.target.Target, thread_nums: int, thread_var: tir.Var):
        """底层转换：生成具体的 TIR 代码"""
        copy_inst = self._select_copy_instruction(target, thread_nums)
        impl_class = self._get_implementation_class(copy_inst, target)
        return impl_class(self).lower(layout_map, target, thread_nums, thread_var)

    def _select_copy_instruction(self, target, thread_nums):
        # 这里可以根据 target.arch 或者 annotations 决定
        # 比如检测是否有 "swizzle_8thread_warp" 标记
        if self.annotations.get("swizzle_8thread_warp"):
            return "swizzled_normal"
        # 默认返回通用异步指令
        return "cp_async"

    def _get_implementation_class(self, inst_type: str, target):
        if inst_type == "swizzled_normal":
            return AsyncCopySwizzled
