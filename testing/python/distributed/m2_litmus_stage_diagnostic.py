"""Narrow normal-pipeline diagnostic for the block litmus (not an acceptance test)."""
from testing.python.distributed.m2_dsl_block_litmus import block_publication
from tilelang.distributed.ipc_binding import bind_ipc, prepare_ipc_primfunc

bound = bind_ipc(block_publication, ["src", "payload", "ready", "ack"])
print("bound", flush=True)
prepared = prepare_ipc_primfunc(bound, target_kind="hip")
print("prepared", len(prepared.params), flush=True)
