"""Standalone M1 normalization and validation for distributed TIR.

This module deliberately stops before device lowering.  It converts the
frontend-only remote-view token into ordinary typed TIR storage and verifies
the conservative M1 contract.  In particular, it never lets an unnormalised
remote view fall through as a local buffer access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tvm
from tvm import ir, tir

from .common import _REMOTE_VIEW_TOKEN, _buffer_type_for_decl, _is_rank_dtype
from .comm import validate_block_copy_regions


class DistributedIRError(ValueError):
    """Raised when M1 distributed TIR is not safe to lower."""


def _op_name(expr: Any) -> str | None:
    if not isinstance(expr, tir.Call) or not isinstance(expr.op, ir.Op):
        return None
    return str(expr.op.name)


def _same(a: Any, b: Any) -> bool:
    return bool(a.same_as(b))


def _remote_token(buffer: tir.Buffer) -> tir.Call | None:
    """Extract exactly one token from a frontend remote-view buffer."""
    if buffer.elem_offset is None:
        return None
    tokens: list[tir.Call] = []

    def visit(node: Any) -> None:
        if _op_name(node) == _REMOTE_VIEW_TOKEN:
            tokens.append(node)

    tir.stmt_functor.post_order_visit(buffer.elem_offset, visit)
    if len(tokens) > 1:
        raise DistributedIRError("remote_view element offset contains more than one remote token")
    return tokens[0] if tokens else None


def _remote_buffers(node: Any) -> list[tir.Buffer]:
    result: list[tir.Buffer] = []

    def append_if_remote(candidate: tir.Buffer) -> None:
        if _remote_token(candidate) is not None and not any(_same(candidate, item) for item in result):
            result.append(candidate)

    def visit(item: Any) -> None:
        if isinstance(item, tir.BufferLoad):
            append_if_remote(item.buffer)
        elif isinstance(item, tir.BufferStore):
            append_if_remote(item.buffer)

    tir.stmt_functor.post_order_visit(node, visit)
    return result


def _region_buffer(expr: Any) -> tir.Buffer | None:
    if _op_name(expr) != "tl.tileop.region" or not expr.args or not isinstance(expr.args[0], tir.BufferLoad):
        return None
    return expr.args[0].buffer


def _integer_peer(peer: Any) -> bool:
    return not isinstance(peer, bool) and (
        isinstance(peer, int) or (isinstance(peer, tir.PrimExpr) and _is_rank_dtype(peer.dtype))
    )


def _int_value(expr: Any) -> int | None:
    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    if isinstance(expr, int) and not isinstance(expr, bool):
        return expr
    return None


def _uint32_value(expr: Any) -> bool:
    if isinstance(expr, bool):
        return False
    value = _int_value(expr)
    if value is not None:
        return 0 <= value <= 0xFFFFFFFF
    return isinstance(expr, tir.PrimExpr) and expr.dtype == "uint32"


def _raw_scalar_dtype(buffer: tir.Buffer | None) -> bool:
    return buffer is not None and buffer.dtype in {"int32", "uint32", "float32"}


def _raw_memory_mode(sem: Any, scope: Any, allowed: set[int]) -> bool:
    return _int_value(scope) in {1, 2} and _int_value(sem) in allowed


def _make_alias(buffer: tir.Buffer, token: tir.Call, ordinal: int) -> tuple[tir.Var, tir.Buffer]:
    if len(token.args) < 3:
        raise DistributedIRError("malformed tl.remote_view_token")
    # The new pointer is explicitly typed.  Its Buffer retains the original
    # view's shape/layout/alignment but resets elem_offset to the base view
    # offset carried in the token.  `remote_address` denotes allocation base,
    # so that offset is added exactly once by normal TIR buffer addressing.
    pointer_type = ir.PointerType(ir.PrimType(buffer.dtype), "global")
    data = tir.Var(f"{buffer.name}_remote_ptr_{ordinal}", pointer_type)
    alias = tir.decl_buffer(
        buffer.shape,
        buffer.dtype,
        name=f"{buffer.name}_remote_alias_{ordinal}",
        data=data,
        strides=buffer.strides,
        elem_offset=token.args[2],
        scope="global",
        data_alignment=buffer.data_alignment,
        offset_factor=buffer.offset_factor,
        buffer_type=_buffer_type_for_decl(buffer),
        axis_separators=buffer.axis_separators,
    )
    return data, alias


@dataclass
class _RemoteAlias:
    original: tir.Buffer
    token: tir.Call
    data: tir.Var
    alias: tir.Buffer

    @property
    def provenance(self) -> tir.PrimExpr:
        return self.token.args[0]

    @property
    def peer(self) -> tir.PrimExpr:
        return self.token.args[1]


class _AliasNormalizer(tir.functor.PyStmtExprMutator):
    """Rewrite remote buffer accesses and bind pointers at their use site."""

    def __init__(self, func: tir.PrimFunc):
        super().__init__()
        self._func = func
        self._next_alias = 0
        self._active_entries: list[list[_RemoteAlias]] = []
        self._access_depth = 0

    def _entry(self, buffer: tir.Buffer) -> _RemoteAlias | None:
        token = _remote_token(buffer)
        if token is None:
            return None
        # A binding may be emitted around every legal access statement.  Each
        # binding therefore owns a fresh typed Var: reusing one Var in several
        # sibling LetStmt scopes violates SSA even if the remote-view identity
        # is the same.
        data, alias = _make_alias(buffer, token, self._next_alias)
        self._next_alias += 1
        entry = _RemoteAlias(buffer, token, data, alias)
        return entry

    def _entries_in(self, node: Any) -> list[_RemoteAlias]:
        entries: list[_RemoteAlias] = []
        buffers = _remote_buffers(node)
        if isinstance(node, tir.BufferStore) and _remote_token(node.buffer) is not None:
            buffers.append(node.buffer)
        for buffer in buffers:
            entry = self._entry(buffer)
            assert entry is not None
            if not any(_same(entry.original, existing.original) for existing in entries):
                entries.append(entry)
        return entries

    def _active_entry(self, buffer: tir.Buffer) -> _RemoteAlias | None:
        for group in reversed(self._active_entries):
            for entry in group:
                if _same(entry.original, buffer):
                    return entry
        return None

    def _wrap(self, stmt: tir.Stmt, entries: list[_RemoteAlias]) -> tir.Stmt:
        # Wrap the smallest statement containing the access.  This matters for
        # peer expressions defined by an enclosing serial loop, LetStmt, or a
        # blockIdx thread binding: a PrimFunc-root binding would leave those
        # variables free.  Per-statement bindings are correct though they are
        # intentionally not the M2 pointer-hoisting optimization.
        for entry in reversed(entries):
            address = tir.call_intrin(
                "handle", tir.op.Op.get("tl.remote_address"), entry.provenance, entry.peer
            )
            stmt = tir.LetStmt(entry.data, address, tir.DeclBuffer(entry.alias, stmt))
        return stmt

    def _rewrite_expr(self, expr: tir.PrimExpr) -> tir.PrimExpr:
        """Rewrite the explicit call/region form emitted by M1 DSL builders.

        The matched TVM revision's generic Python mutator reconstructs a Call
        with an undefined optional field while visiting opaque TileLang calls.
        M1's supported statement-local uses are deliberately explicit TIR
        Calls and BufferLoads, so rebuilding those two concrete forms avoids
        that ABI issue without changing or weakening the accepted IR.
        """
        if isinstance(expr, tir.BufferLoad):
            entry = self._active_entry(expr.buffer)
            if _remote_token(expr.buffer) is None:
                return expr
            if entry is None:
                raise DistributedIRError("remote_view access escaped its statement-local binding")
            return tir.BufferLoad(entry.alias, list(expr.indices), expr.predicate)
        if isinstance(expr, tir.Call):
            args = [self._rewrite_expr(arg) for arg in expr.args]
            return tir.Call(str(expr.dtype), expr.op, args)
        if _has_remote_load(expr):
            raise DistributedIRError(
                "remote_view expression is supported only through explicit M1 TIR calls/regions"
            )
        return expr

    def visit_buffer_load_(self, op: tir.BufferLoad) -> tir.PrimExpr:
        if _remote_token(op.buffer) is None:
            return super().visit_buffer_load_(op)
        entry = self._active_entry(op.buffer)
        if self._access_depth == 0:
            raise DistributedIRError(
                "remote_view is supported only in an Evaluate/BufferStore access statement in M1; "
                "a remote load in a condition or Let value has no legal pointer binding scope"
            )
        if entry is None:
            raise DistributedIRError("remote_view access escaped its statement-local binding")
        indices = [self.visit_expr(index) for index in op.indices]
        predicate = self.visit_expr(op.predicate) if op.predicate is not None else None
        return tir.BufferLoad(entry.alias, indices, predicate)

    def visit_buffer_store_(self, op: tir.BufferStore) -> tir.Stmt:
        entries = self._entries_in(op)
        if not entries:
            return op
        self._active_entries.append(entries)
        try:
            value = self._rewrite_expr(op.value)
            indices = [self._rewrite_expr(index) for index in op.indices]
            predicate = self._rewrite_expr(op.predicate) if op.predicate is not None else None
        finally:
            self._active_entries.pop()
        entry = next((item for item in entries if _same(item.original, op.buffer)), None)
        rewritten = tir.BufferStore(entry.alias if entry else op.buffer, value, indices, predicate)
        return self._wrap(rewritten, entries)

    def visit_evaluate_(self, op: tir.Evaluate) -> tir.Stmt:
        entries = self._entries_in(op.value)
        if not entries:
            return op
        self._active_entries.append(entries)
        try:
            value = self._rewrite_expr(op.value)
        finally:
            self._active_entries.pop()
        return self._wrap(tir.Evaluate(value), entries)


_AliasNormalizer = tir.functor.mutator(_AliasNormalizer)


def _collect_bindings(
    func: tir.PrimFunc,
) -> tuple[
    list[tir.Var],
    dict[tir.Var, tir.PrimExpr],
    dict[tir.Var, tuple[tir.PrimExpr, tir.PrimExpr, tir.PrimExpr]],
    set[tir.Var],
    set[tir.Var],
]:
    """Collect identity-based binding facts used by the conservative verifier."""
    scalar_params: list[tir.Var] = []
    for param in func.params:
        if isinstance(param, tir.Var) and _is_rank_dtype(param.dtype):
            scalar_params.append(param)
    lets: dict[tir.Var, tir.PrimExpr] = {}
    serial_vars: dict[tir.Var, tuple[tir.PrimExpr, tir.PrimExpr, tir.PrimExpr]] = {}
    block_vars: set[tir.Var] = set()
    thread_vars: set[tir.Var] = set()

    def visit(node: Any) -> None:
        if isinstance(node, tir.LetStmt):
            lets[node.var] = node.value
        elif isinstance(node, tir.For):
            if node.kind in (tir.ForKind.SERIAL, tir.ForKind.UNROLLED):
                # Modern TIR may expose an explicit step.  If absent, its
                # semantic step is one.  A thread-varying step invalidates a
                # serial-loop peer just like a varying min/extent would.
                step = getattr(node, "step", None)
                if step is None:
                    step = tir.const(1, node.min.dtype)
                serial_vars[node.loop_var] = (node.min, node.extent, step)
            elif node.kind == tir.ForKind.THREAD_BINDING:
                tag = str(getattr(node.thread_binding, "thread_tag", ""))
                (block_vars if tag.startswith("blockIdx") else thread_vars).add(node.loop_var)
        elif isinstance(node, tir.AttrStmt) and node.attr_key == "thread_extent":
            candidate = getattr(node.node, "var", None)
            if isinstance(candidate, tir.Var):
                tag = str(getattr(node.node, "thread_tag", ""))
                (block_vars if tag.startswith("blockIdx") else thread_vars).add(candidate)

    tir.stmt_functor.post_order_visit(func.body, visit)
    return scalar_params, lets, serial_vars, block_vars, thread_vars


def _contains(items: list[tir.Var] | set[tir.Var], value: tir.Var) -> bool:
    return any(_same(item, value) for item in items)


def _is_block_uniform(
    peer: tir.PrimExpr,
    scalar_params: list[tir.Var],
    lets: dict[tir.Var, tir.PrimExpr],
    serial_vars: dict[tir.Var, tuple[tir.PrimExpr, tir.PrimExpr, tir.PrimExpr]],
    block_vars: set[tir.Var],
    thread_vars: set[tir.Var],
) -> bool:
    """Return whether an M1 peer expression has one value per block.

    This is deliberately a whitelist.  It accepts scalar ABI parameters,
    rank/world-size, blockIdx and serial loop variables (and arithmetic/Let
    aliases of them).  ThreadIdx, buffer loads and unfamiliar calls are
    rejected rather than guessed to be uniform.
    """
    seen: list[tir.Var] = []

    def uniform_var(var: tir.Var) -> bool:
        if any(_same(var, item) for item in seen):
            return False
        if _contains(thread_vars, var):
            return False
        if _contains(block_vars, var) or _contains(scalar_params, var):
            return True
        for loop_var, bounds in serial_vars.items():
            if _same(loop_var, var):
                return uniform(bounds[0]) and uniform(bounds[1]) and uniform(bounds[2])
        for bound, value in lets.items():
            if _same(bound, var):
                seen.append(var)
                result = uniform(value)
                seen.pop()
                return result
        return False

    def uniform(expr: Any) -> bool:
        valid = True

        def visit(node: Any) -> None:
            nonlocal valid
            if not valid:
                return
            if isinstance(node, tir.Var):
                valid = uniform_var(node)
            elif isinstance(node, tir.BufferLoad):
                valid = False
            elif isinstance(node, tir.Call):
                name = _op_name(node)
                if name not in ("tl.get_rank", "tl.get_num_ranks"):
                    valid = False

        tir.stmt_functor.post_order_visit(expr, visit)
        return valid

    return uniform(peer)


def _verify_uniform_peers(func: tir.PrimFunc) -> None:
    scalar_params, lets, serial_vars, block_vars, thread_vars = _collect_bindings(func)
    failures: list[str] = []

    def verify_buffer(buffer: tir.Buffer) -> None:
        token = _remote_token(buffer)
        if token is not None:
            if not _integer_peer(token.args[1]):
                failures.append(f"remote_view peer for {buffer.name} must be an integer rank")
            elif not _is_block_uniform(
                token.args[1], scalar_params, lets, serial_vars, block_vars, thread_vars
            ):
                failures.append(f"remote_view peer for {buffer.name} is not block-uniform")

    def visit(node: Any) -> None:
        if isinstance(node, tir.BufferLoad):
            verify_buffer(node.buffer)
        elif isinstance(node, tir.BufferStore):
            verify_buffer(node.buffer)
        elif _op_name(node) in {
            "tl.signal_set", "tl.tileop.wait", "tl.tileop.put_block", "tl.tileop.get_block",
            "tl.tileop.st", "tl.tileop.ld", "tl.remote_address",
        }:
            name = _op_name(node)
            # wait is `(relation, region, expected, peer, scope, semantics)`;
            # its last argument is semantics, not rank.
            peer_index = 3 if name == "tl.tileop.wait" else -1
            if len(node.args) < (4 if peer_index == 3 else 1):
                failures.append(f"{name} has no peer argument")
                return
            peer = node.args[peer_index]
            if not _integer_peer(peer):
                failures.append(f"{_op_name(node)} peer must be an integer rank")
            elif not _is_block_uniform(peer, scalar_params, lets, serial_vars, block_vars, thread_vars):
                failures.append(f"{_op_name(node)} peer is not block-uniform")

    tir.stmt_functor.post_order_visit(func.body, visit)

    # A block copy distributes its element loop over every participating
    # thread.  A peer expression can be block-uniform while the *statement*
    # is guarded by threadIdx.x; in that case silently executing lane zero
    # would leave the rest of the range stale.  Keep signal/wait flexible for
    # the documented thread-zero publication pattern, but reject a
    # cooperative put/get under a thread-varying conditional.
    def control_is_block_uniform(expr: tir.PrimExpr) -> bool:
        """Prove a control predicate/loop bound is identical for a CTA.

        This deliberately uses the same Let-aware whitelist as peer
        validation.  Looking only for a raw threadIdx Var misses both
        ``let gate = threadIdx.x`` and BufferLoad-mediated predicates, which
        would make a real `__syncthreads()` deadlock after M2 lowering.
        """
        return _is_block_uniform(
            expr, scalar_params, lets, serial_vars, block_vars, thread_vars
        )

    def guarded_copy(stmt: tir.Stmt | None) -> bool:
        if stmt is None:
            return False
        found = False

        def find_copy(node: Any) -> None:
            nonlocal found
            if _op_name(node) in {"tl.tileop.put_block", "tl.tileop.get_block"}:
                found = True

        tir.stmt_functor.post_order_visit(stmt, find_copy)
        return found

    def guarded_block_barrier(stmt: tir.Stmt | None) -> bool:
        """Recognize the CTA sync that M2 turns into an IPC barrier."""
        if stmt is None:
            return False
        found = False

        def find_barrier(node: Any) -> None:
            nonlocal found
            if _op_name(node) != "tir.tvm_storage_sync" or not node.args:
                return
            scope = node.args[0]
            if isinstance(scope, tir.StringImm) and str(scope.value) in {"shared", "shared.dyn"}:
                found = True

        tir.stmt_functor.post_order_visit(stmt, find_barrier)
        return found

    def guarded_collective(stmt: tir.Stmt | None) -> bool:
        return stmt is not None and (guarded_copy(stmt) or guarded_block_barrier(stmt))

    def for_controls_are_uniform(node: tir.For) -> bool:
        step = getattr(node, "step", None)
        if step is None:
            step = tir.const(1, node.min.dtype)
        return all(control_is_block_uniform(value) for value in (node.min, node.extent, step))

    def find_divergent_collective(node: Any) -> None:
        if isinstance(node, tir.IfThenElse) and not control_is_block_uniform(node.condition):
            if guarded_copy(node.then_case) or guarded_copy(node.else_case):
                failures.append("cooperative put_block/get_block must not be guarded by a thread-varying condition")
            if guarded_block_barrier(node.then_case) or guarded_block_barrier(node.else_case):
                failures.append("IPC CTA barrier must not be guarded by a thread-varying condition")
        elif isinstance(node, tir.For) and guarded_collective(node.body) and not for_controls_are_uniform(node):
            failures.append("cooperative work and IPC CTA barriers require block-uniform For controls")
        elif isinstance(node, tir.While) and guarded_collective(node.body) and not control_is_block_uniform(node.condition):
            failures.append("cooperative work and IPC CTA barriers require a block-uniform While condition")

    tir.stmt_functor.post_order_visit(func.body, find_divergent_collective)
    if failures:
        raise DistributedIRError("; ".join(failures))


def assert_remote_views_normalized(func: tir.PrimFunc) -> None:
    """Reject frontend remote-view tokens before lowering/code generation."""
    remaining: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, tir.BufferLoad) and _remote_token(node.buffer) is not None:
            remaining.append(str(node.buffer.name))
        elif isinstance(node, tir.BufferStore) and _remote_token(node.buffer) is not None:
            remaining.append(str(node.buffer.name))

    tir.stmt_functor.post_order_visit(func.body, visit)
    if remaining:
        raise DistributedIRError(
            "unnormalized remote_view cannot execute as local memory: " + ", ".join(remaining)
        )


def _has_remote_load(expr: tir.PrimExpr) -> bool:
    """Whether an expression loads an unnormalised remote buffer."""
    found = False

    def visit(node: Any) -> None:
        nonlocal found
        if isinstance(node, tir.BufferLoad) and _remote_token(node.buffer) is not None:
            found = True

    tir.stmt_functor.post_order_visit(expr, visit)
    return found


def _reject_unbindable_remote_loads(stmt: tir.Stmt) -> None:
    """Fail closed before the FFI mutator sees an unsupported use position.

    A statement-local remote pointer binding dominates an Evaluate or a
    BufferStore, but not a Let/If condition used to choose that statement.
    Check those expression slots in ordinary Python so callers receive the
    documented DistributedIRError rather than an FFI callback wrapper.
    """
    unsupported: list[str] = []

    def visit(node: Any) -> None:
        values: list[Any] = []
        if isinstance(node, tir.LetStmt):
            values.append(node.value)
        elif isinstance(node, tir.IfThenElse):
            values.append(node.condition)
        elif isinstance(node, tir.AssertStmt):
            values.extend((node.condition, node.message))
        elif isinstance(node, tir.For):
            values.extend((node.min, node.extent, getattr(node, "step", None)))
        for value in values:
            if isinstance(value, tir.PrimExpr) and _has_remote_load(value):
                unsupported.append(type(node).__name__)

    tir.stmt_functor.post_order_visit(stmt, visit)
    if unsupported:
        raise DistributedIRError(
            "remote_view load has no legal pointer binding scope in M1 "
            f"({', '.join(unsupported)})"
        )


def normalize_remote_views(func: tir.PrimFunc) -> tir.PrimFunc:
    """Normalize typed frontend remote views into bound TIR aliases.

    `remote=False` creates no token and is unchanged.  The returned PrimFunc
    contains a unique typed pointer/DeclBuffer alias for every remote view
    identity and explicit `tl.remote_address(provenance, peer)` bindings.
    """
    if not isinstance(func, tir.PrimFunc):
        raise TypeError("normalize_remote_views expects a tir.PrimFunc")
    _verify_uniform_peers(func)
    _reject_unbindable_remote_loads(func.body)
    normalizer = _AliasNormalizer(func)
    result = func.with_body(normalizer.visit_stmt(func.body))
    assert_remote_views_normalized(result)
    return result


def verify_distributed_ir(func: tir.PrimFunc, *, require_normalized: bool = True) -> None:
    """Validate M1-side type, peer-uniformity and normalization invariants."""
    if not isinstance(func, tir.PrimFunc):
        raise TypeError("verify_distributed_ir expects a tir.PrimFunc")
    _verify_uniform_peers(func)
    if require_normalized:
        assert_remote_views_normalized(func)

    invalid: list[str] = []

    def visit(node: Any) -> None:
        name = _op_name(node)
        if name == "tl.remote_address":
            if len(node.args) != 2 or _op_name(node.args[0]) != "tl.tileop.region":
                invalid.append("remote_address must retain tl.region(base) provenance and peer")
        elif name == "tl.tileop.wait":
            if len(node.args) != 6:
                invalid.append("wait must have relation, region, expected, peer, scope and semantics")
            else:
                flag = _region_buffer(node.args[1])
                if flag is None or flag.dtype != "uint32":
                    invalid.append("wait requires a uint32 tl.region flag")
                if not _uint32_value(node.args[2]):
                    invalid.append("wait expected must be uint32-representable")
                if _int_value(node.args[0]) not in (0, 2):
                    invalid.append("M1 wait relation must be EQ(0) or GE(2)")
                if _int_value(node.args[4]) != 0 or _int_value(node.args[5]) != 0:
                    invalid.append("M1 wait only supports SYS/ACQUIRE")
        elif name == "tl.signal_set":
            flag = _region_buffer(node.args[0]) if len(node.args) == 3 else None
            if flag is None or flag.dtype != "uint32":
                invalid.append("signal_set requires a uint32 tl.region flag")
            elif not _uint32_value(node.args[1]):
                invalid.append("signal_set value must be uint32-representable")
        elif name in {"tl.tileop.put_block", "tl.tileop.get_block"}:
            if len(node.args) != 4:
                invalid.append(f"{name} must have source region, destination region, size and peer")
            else:
                try:
                    validate_block_copy_regions(node.args[0], node.args[1], node.args[2])
                except (TypeError, ValueError) as err:
                    invalid.append(f"{name}: {err}")
        elif name == "tl.tileop.ld":
            src, dst = (_region_buffer(node.args[0]), _region_buffer(node.args[1])) if len(node.args) == 5 else (None, None)
            if not _raw_scalar_dtype(src) or not _raw_scalar_dtype(dst) or src.dtype != dst.dtype:
                invalid.append("ld requires same-dtype HIP-M1 scalar read and writable regions")
            elif not _raw_memory_mode(node.args[2], node.args[3], {0, 1, 2, 4}):
                invalid.append("ld has unsupported HIP-M1 scope/semantics")
        elif name == "tl.tileop.st":
            dst = _region_buffer(node.args[0]) if len(node.args) == 5 else None
            if not _raw_scalar_dtype(dst):
                invalid.append("st requires a HIP-M1 scalar writable region")
            elif not (isinstance(node.args[1], tir.PrimExpr) and node.args[1].dtype == dst.dtype):
                invalid.append("st value dtype must match destination")
            elif not _raw_memory_mode(node.args[2], node.args[3], {0, 1, 3, 4}):
                invalid.append("st has unsupported HIP-M1 scope/semantics")

    tir.stmt_functor.post_order_visit(func.body, visit)
    if invalid:
        raise DistributedIRError("; ".join(invalid))
