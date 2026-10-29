"""CPU TIR checks for the explicit persistent DCU resource scope.

The valid case lowers the public compute-only example rather than a synthetic
copy-only kernel: marker acceptance therefore still requires an actual GEMM K
loop.  The small copy-only probes below only check rejection/fallback policy;
they must not be used to broaden the persistent-GEMM entry contract.
"""
import importlib.util
from pathlib import Path

import pytest

import tilelang
from tilelang import tvm
from tvm.script import tir as T


ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "examples" / "distributed" / "gemm_rs_prototype" / "gemm_persistent_resource.py"
TARGET = "hip -mcpu=gfx936"


def _load_example():
    spec = importlib.util.spec_from_file_location("persistent_resource_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _var_name(var) -> str:
    return var.name if hasattr(var, "name") else var.name_hint


def _free_vars(expr):
    variables = set()

    def visit(node):
        if isinstance(node, tvm.tir.Var):
            variables.add(node)

    tvm.tir.stmt_functor.post_order_visit(expr, visit)
    return variables


def _marker_value(for_node) -> bool:
    value = for_node.annotations.get("tl.dcu_persistent_tile")
    return value == 1 or getattr(value, "value", None) == 1


def _collect_let_definitions(stmt):
    definitions = {}

    def visit(node):
        if isinstance(node, tvm.tir.LetStmt):
            definitions[node.var] = node.value

    tvm.tir.stmt_functor.post_order_visit(stmt, visit)
    return definitions


def _depends_on(expr, targets, definitions, seen=None) -> bool:
    """Follow TIR Let Vars by identity, not by their printed names."""
    seen = set() if seen is None else seen
    for var in _free_vars(expr):
        if var in targets:
            return True
        if var in definitions and var not in seen:
            if _depends_on(definitions[var], targets, definitions, seen | {var}):
                return True
    return False


def _resource_scopes(stmt):
    """Return resource Var -> (base, marker-loop Vars, blockIdx Vars)."""
    scopes = {}

    def visit(current, marker_vars, block_vars):
        if isinstance(current, tvm.tir.AttrStmt):
            node = current.node
            tag = getattr(node, "thread_tag", "")
            extra = {node.var} if str(tag).startswith("blockIdx") else set()
            visit(current.body, marker_vars, block_vars | extra)
            return
        if isinstance(current, tvm.tir.For):
            marker_vars = marker_vars | ({current.loop_var} if _marker_value(current) else set())
            binding = current.thread_binding
            tag = getattr(binding, "thread_tag", "") if binding is not None else ""
            block_vars = block_vars | ({current.loop_var} if str(tag).startswith("blockIdx") else set())
            visit(current.body, marker_vars, block_vars)
            return
        if isinstance(current, tvm.tir.LetStmt):
            if isinstance(current.value, tvm.tir.Call) and current.value.op.name == "tl.make_dcu_resource":
                scopes[current.var] = (current.value.args[1], set(marker_vars), set(block_vars))
            visit(current.body, marker_vars, block_vars)
            return
        if isinstance(current, tvm.tir.SeqStmt):
            for child in current.seq:
                visit(child, marker_vars, block_vars)
            return
        if isinstance(current, tvm.tir.IfThenElse):
            visit(current.then_case, marker_vars, block_vars)
            if current.else_case is not None:
                visit(current.else_case, marker_vars, block_vars)
            return
        if isinstance(current, tvm.tir.Allocate):
            visit(current.body, marker_vars, block_vars)
            return
        if isinstance(current, tvm.tir.DeclBuffer):
            visit(current.body, marker_vars, block_vars)
            return
        if isinstance(current, tvm.tir.AssertStmt):
            visit(current.body, marker_vars, block_vars)
            return
        # The final lowered GEMM body normally reaches all copies through the
        # cases above.  Post-order collection below still sees copies in any
        # other statement wrapper; those must have a recorded resource scope.

    visit(stmt, set(), set())
    return scopes


def _resources_and_offsets(func):
    """Return source -> (resource-var, base, copy coordinate expressions)."""
    resources, copies = {}, {}

    def visit(node):
        if isinstance(node, tvm.tir.LetStmt) and isinstance(node.value, tvm.tir.Call):
            if node.value.op.name == "tl.make_dcu_resource":
                resources[node.var] = (_var_name(node.value.args[0]), node.value.args[1])
        if isinstance(node, tvm.tir.Call) and node.op.name in {
            "tl.dcu_async_copy", "tl.dcu_async_copy_idxen",
        }:
            # arg3 is the normal resource-relative source offset.  idxen has
            # a second coordinate at arg4; it is the critical A-row path.
            coords = [node.args[3]]
            if node.op.name == "tl.dcu_async_copy_idxen":
                coords.append(node.args[4])
            copies.setdefault(node.args[2], []).append(coords)

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return {
        source: (resource, base, copies[resource])
        for resource, (source, base) in resources.items()
        if resource in copies
    }


@pytest.mark.parametrize(
    ("blocks", "b_depends_on_marker"),
    [(15, True), (16, False)],
    ids=["both-operands-vary", "b-base-constant-within-tile-loop"],
)
def test_persistent_gemm_scopes_a_b_resources_and_emits_local_offsets(
    blocks, b_depends_on_marker,
):
    example = _load_example()
    with tvm.target.Target(TARGET):
        artifact = tilelang.lower(
            example.persistent_gemm(2048, 2048, 512, blocks=blocks),
            target=TARGET, enable_device_compile=False, enable_host_codegen=False,
        )
    device = next(value for value in artifact.device_mod.functions.values()
                  if isinstance(value, tvm.tir.PrimFunc))
    definitions = _collect_let_definitions(device.body)
    scopes = _resource_scopes(device.body)
    resources = _resources_and_offsets(device)

    # A/B retain the existing uppercase operand spelling used by idxen; both
    # receive a tile-local resource in the real four-stage GEMM lowering.
    assert {"A", "B"} <= set(resources)
    for source in ("A", "B"):
        resource, base, coordinate_lists = resources[source]
        assert resource in scopes, (source, resource)
        scoped_base, marker_vars, block_vars = scopes[resource]
        assert marker_vars, (source, "resource was hoisted outside marked tile loop")
        assert scoped_base.same_as(base)
        # A must vary with every output-tile traversal.  B may legitimately
        # be constant across a particular loop's iterations: with nTiles=8
        # and blocks=16, (block + 16*iteration) % 8 is block % 8.  It is still
        # scoped inside the marker and its emitted copy coordinates must stay
        # local.  blocks=15 covers the complementary case where both bases
        # vary, so this test does not encode either shape as a blanket rule.
        depends_on_marker = _depends_on(base, marker_vars, definitions)
        assert depends_on_marker is (source == "A" or b_depends_on_marker), (source, base)
        # Identity plus Let expansion catches anonymous lowered loop vars such
        # as ``_tmp``/``bx`` that a printed-name check would miss.  Check both
        # normal arg3 and idxen's separate row arg4.
        forbidden = marker_vars | block_vars
        for coordinates in coordinate_lists:
            for coordinate in coordinates:
                assert not _depends_on(coordinate, forbidden, definitions), (
                    source, base, coordinate,
                )


@T.prim_func
def _unmarked_copy_fallback(A: T.Buffer((8192,), "float16")):
    shared = T.alloc_buffer((256,), "float16", scope="shared")
    tx = T.launch_thread("threadIdx.x", 8)
    for tile in T.serial(4):
        with T.attr(0, "async_scope", 1):
            shared[T.Ramp(tx * 8, 1, 8)] = A[T.Ramp(tile * 256 + tx * 8, 1, 8)]


@T.prim_func
def _marked_non_gemm_is_rejected(A: T.Buffer((8192,), "float16")):
    shared = T.alloc_buffer((256,), "float16", scope="shared")
    tx = T.launch_thread("threadIdx.x", 8)
    for tile in T.serial(4, annotations={"tl.dcu_persistent_tile": 1}):
        with T.attr(0, "async_scope", 1):
            shared[T.Ramp(tx * 8, 1, 8)] = A[T.Ramp(tile * 256 + tx * 8, 1, 8)]


@T.prim_func
def _bad_marker_value(A: T.Buffer((8192,), "float16")):
    # The frontend contract is exact integer 1, not a generic truthy flag.
    for tile in T.serial(4, annotations={"tl.dcu_persistent_tile": 2}):
        A[tile] = A[tile]


@T.prim_func
def _thread_dependent_marker_is_rejected(A: T.Buffer((8192,), "float16")):
    shared = T.alloc_buffer((256,), "float16", scope="shared")
    tx = T.launch_thread("threadIdx.x", 8)
    if tx == 0:
        for tile in T.serial(4, annotations={"tl.dcu_persistent_tile": 1}):
            with T.attr(0, "async_scope", 1):
                shared[T.Ramp(tx * 8, 1, 8)] = A[T.Ramp(tile * 256 + tx * 8, 1, 8)]


@T.prim_func
def _impure_marker_control_is_rejected(A: T.Buffer((8192,), "float16")):
    shared = T.alloc_buffer((256,), "float16", scope="shared")
    tx = T.launch_thread("threadIdx.x", 8)
    if A[0] != T.float16(0):
        for tile in T.serial(4, annotations={"tl.dcu_persistent_tile": 1}):
            with T.attr(0, "async_scope", 1):
                shared[T.Ramp(tx * 8, 1, 8)] = A[T.Ramp(tile * 256 + tx * 8, 1, 8)]


@T.prim_func
def _outer_impure_offset_alias_is_rejected(A: T.Buffer((8192,), "float16")):
    shared = T.alloc_buffer((256,), "float16", scope="shared")
    tx = T.launch_thread("threadIdx.x", 8)
    # This Let dominates the marker but is not a legal cached-base component.
    # It directly tests recursive purity checking beyond lets in loop.body.
    alias = T.call_extern("int32", "test_impure_offset_alias")
    for tile in T.serial(4, annotations={"tl.dcu_persistent_tile": 1}):
        with T.attr(0, "async_scope", 1):
            shared[T.Ramp(tx * 8, 1, 8)] = A[T.Ramp(alias + tile * 256 + tx * 8, 1, 8)]


def test_unmarked_copy_keeps_legacy_fallback_and_marker_does_not_enable_non_gemm():
    fallback = tilelang.transform.LowerSharedGlobalCopy()(
        tvm.IRModule({"main": _unmarked_copy_fallback})
    )["main"].script()
    assert "T.make_dcu_resource(A" in fallback

    with pytest.raises(tvm.error.TVMError, match="innermost DCU GEMM K loop"):
        tilelang.transform.LowerSharedGlobalCopy()(
            tvm.IRModule({"main": _marked_non_gemm_is_rejected})
        )


def test_persistent_marker_rejects_malformed_and_divergent_control_scopes():
    lower = tilelang.transform.LowerSharedGlobalCopy()
    with pytest.raises(tvm.error.TVMError, match="integer value 1"):
        lower(tvm.IRModule({"main": _bad_marker_value}))
    for func in (_thread_dependent_marker_is_rejected, _impure_marker_control_is_rejected):
        with pytest.raises(tvm.error.TVMError, match="thread-dependent or impure"):
            lower(tvm.IRModule({"main": func}))
    with pytest.raises(tvm.error.TVMError, match="contains a load or call"):
        lower(tvm.IRModule({"main": _outer_impure_offset_alias_is_rejected}))
