"""Four-PE, all-direction M2 block-publication litmus built from public DSL.

Each of four ranks publishes three unequal typed runs to the other ranks:
12 directed edges total.  Every writer fences after its own cooperative store,
then the CTA rendezvouses before thread zero releases the peer ready flag.
Consumers first issue volatile old-value reads, have thread zero acquire-wait,
CTA-rendezvous, weak-read the complete tile, and return an ACK before the next
in-kernel epoch.  The host runs multiple collective reset batches with distinct
seeds, and validates every epoch plus every payload slot tail and guard.
"""
import ctypes
import os
import time

import tilelang
from tilelang import language as T
from tilelang.distributed.ipc_binding import bind_ipc

# The ctypes backend's persistent host-source cache is not part of M2 (and
# this backend reports no host source).  Compile the public DSL fresh so a
# previous process's temporary artifact cannot mask this protocol run.
tilelang.disable_cache()

THREADS, SLOT, GUARD = 256, 256, 64
# Keep the default quick enough for development, while letting a reviewer
# amplify the same public DSL protocol without changing its source.
EPOCHS = int(os.environ.get("M2_LITMUS_EPOCHS", "3"))
BATCHES = int(os.environ.get("M2_LITMUS_BATCHES", "2"))
if EPOCHS <= 0 or BATCHES <= 0:
    raise ValueError("M2_LITMUS_EPOCHS and M2_LITMUS_BATCHES must be positive")
L0, L1, L2 = 191, 223, 251
TOTAL = 3 * SLOT + GUARD
OBSERVED_TOTAL = EPOCHS * TOTAL
CANARY_SRC_BYTE, CANARY_PAYLOAD_BYTE = 0x3C, 0xA5
CANARY_STALE_BYTE, CANARY_OBSERVED_BYTE = 0x5A, 0xC3
CANARY_SRC = CANARY_SRC_BYTE * 0x01010101
CANARY_PAYLOAD = CANARY_PAYLOAD_BYTE * 0x01010101
CANARY_STALE = CANARY_STALE_BYTE * 0x01010101
CANARY_OBSERVED = CANARY_OBSERVED_BYTE * 0x01010101


@T.prim_func
def block_publication(src: T.Tensor((TOTAL,), "uint32"),
                      payload: T.Tensor((TOTAL,), "uint32"),
                      ready: T.Tensor((3,), "uint32"), ack: T.Tensor((3,), "uint32"),
                      stale: T.Tensor((TOTAL,), "uint32"),
                      observed: T.Tensor((OBSERVED_TOTAL,), "uint32"), seed: T.int32):
    with T.Kernel(1, threads=THREADS):
        tx = T.get_thread_binding()
        rank, world = T.get_rank(), T.get_num_ranks()
        peer0, peer1, peer2 = (rank + 1) % world, (rank + 2) % world, (rank + 3) % world
        from0, from1, from2 = (rank + 3) % world, (rank + 2) % world, (rank + 1) % world
        remote_payload0 = T.remote_view(payload, peer0)
        remote_payload1 = T.remote_view(payload, peer1)
        remote_payload2 = T.remote_view(payload, peer2)
        remote_ready0 = T.remote_view(ready, peer0)
        remote_ready1 = T.remote_view(ready, peer1)
        remote_ready2 = T.remote_view(ready, peer2)
        remote_ack0 = T.remote_view(ack, from0)
        remote_ack1 = T.remote_view(ack, from1)
        remote_ack2 = T.remote_view(ack, from2)
        for epoch in T.serial(1, EPOCHS + 1):
            # Make the last wave deliberately late before it writes its own
            # elements; fence_sys is thread-local and is not a CTA publish.
            if tx >= 192:
                for _ in T.serial(16):
                    T.fence_sys()
            # A reproducible uint32 mix avoids an unrealistically regular
            # payload while retaining an exact host-side oracle.  The key is
            # small for the supported stress ranges; the unsigned multiply
            # supplies the intentional modulo-2^32 mixing.
            key0 = T.cast(seed * 104729 + epoch * 1009 + rank * 97 + tx, "uint32")
            key1 = T.cast(seed * 104729 + epoch * 1009 + rank * 97 + 31 + tx, "uint32")
            key2 = T.cast(seed * 104729 + epoch * 1009 + rank * 97 + 62 + tx, "uint32")
            T.st(src[tx], key0 * T.cast(1664525, "uint32") + T.cast(1013904223, "uint32"), sem="weak")
            T.st(src[SLOT + tx], key1 * T.cast(1664525, "uint32") + T.cast(1013904223, "uint32"), sem="weak")
            T.st(src[2 * SLOT + tx], key2 * T.cast(1664525, "uint32") + T.cast(1013904223, "uint32"), sem="weak")
            # Consumers re-read previously cached payload before the acquire.
            T.ld(payload[tx], stale[tx], sem="volatile")
            T.ld(payload[SLOT + tx], stale[SLOT + tx], sem="volatile")
            T.ld(payload[2 * SLOT + tx], stale[2 * SLOT + tx], sem="volatile")
            T.put_block(src[0], remote_payload0[0], L0, peer=peer0)
            T.put_block(src[SLOT], remote_payload1[SLOT], L1, peer=peer1)
            T.put_block(src[2 * SLOT], remote_payload2[2 * SLOT], L2, peer=peer2)
            # Every participating writer executed its fence; the barrier is
            # mandatory before thread zero performs the release publication.
            T.fence_sys()
            T.sync_threads()
            if tx == 0:
                T.signal_set(remote_ready0[0], T.cast(epoch, "uint32"), peer=peer0)
                T.signal_set(remote_ready1[1], T.cast(epoch, "uint32"), peer=peer1)
                T.signal_set(remote_ready2[2], T.cast(epoch, "uint32"), peer=peer2)
                T.wait_eq(ready[0], T.cast(epoch, "uint32"))
                T.wait_eq(ready[1], T.cast(epoch, "uint32"))
                T.wait_eq(ready[2], T.cast(epoch, "uint32"))
            T.sync_threads()
            # wait_eq on thread zero is the sole system acquire.  The CTA
            # barrier must make that publication visible to all waves; these
            # payload reads deliberately remain weak rather than hiding an
            # invalidation failure behind one acquire per thread.
            T.ld(payload[tx], observed[(epoch - 1) * TOTAL + tx], sem="weak")
            T.ld(payload[SLOT + tx], observed[(epoch - 1) * TOTAL + SLOT + tx], sem="weak")
            T.ld(payload[2 * SLOT + tx], observed[(epoch - 1) * TOTAL + 2 * SLOT + tx], sem="weak")
            T.sync_threads()
            if tx == 0:
                T.signal_set(remote_ack0[0], T.cast(epoch, "uint32"), peer=from0)
                T.signal_set(remote_ack1[1], T.cast(epoch, "uint32"), peer=from1)
                T.signal_set(remote_ack2[2], T.cast(epoch, "uint32"), peer=from2)
                T.wait_eq(ack[0], T.cast(epoch, "uint32"))
                T.wait_eq(ack[1], T.cast(epoch, "uint32"))
                T.wait_eq(ack[2], T.cast(epoch, "uint32"))
            T.sync_threads()


def _hip():
    h = ctypes.CDLL("libamdhip64.so")
    h.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    h.hipMemcpy.restype = ctypes.c_int
    h.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    h.hipMemset.restype = ctypes.c_int
    h.hipEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    h.hipEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    h.hipEventQuery.argtypes = [ctypes.c_void_p]
    h.hipEventDestroy.argtypes = [ctypes.c_void_p]
    return h


def _check(code, what):
    if code:
        raise RuntimeError(f"{what}: HIP status {code}")


def _copy(api, dst, src, nbytes, kind):
    _check(api.hipMemcpy(ctypes.c_void_p(dst), ctypes.c_void_p(src), nbytes, kind), "hipMemcpy")


def _payload_value(seed, epoch, rank, edge, lane):
    key = seed * 104729 + epoch * 1009 + rank * 97 + edge * 31 + lane
    return (key * 1664525 + 1013904223) & 0xFFFFFFFF


if os.environ.get("M2_LITMUS_STAGE") == "prepare":
    from tilelang.distributed.ipc_binding import prepare_ipc_primfunc
    staged = bind_ipc(block_publication, ["src", "payload", "ready", "ack"])
    print("m2-block-litmus-bound", flush=True)
    staged = prepare_ipc_primfunc(staged, target_kind="hip")
    print("m2-block-litmus-prepared", len(staged.params), flush=True)
    raise SystemExit(0)
if os.environ.get("M2_LITMUS_STAGE") == "normalize":
    from tilelang.language.distributed.ir import normalize_remote_views, verify_distributed_ir
    staged = normalize_remote_views(block_publication)
    verify_distributed_ir(staged)
    print("m2-block-litmus-normalized", flush=True)
    raise SystemExit(0)
if os.environ.get("M2_LITMUS_COMPILE_ONLY") == "1":
    compiled = tilelang.compile(bind_ipc(block_publication, ["src", "payload", "ready", "ack"]),
                                target="hip -mcpu=gfx936", execution_backend="ctypes")
    print("m2-block-litmus-compile-ok", len(compiled.artifact.params), flush=True)
    raise SystemExit(0)


mpi = ctypes.CDLL("libmpi.so")
provided = ctypes.c_int()
_check(mpi.MPI_Init_thread(None, None, 1, ctypes.byref(provided)), "MPI_Init_thread")
from tilelang.distributed.ipc_runtime import IPCContext

ctx = IPCContext()
assert ctx.world_size == 4
allocations = [ctx.alloc(name, (OBSERVED_TOTAL if name == "observed" else TOTAL) * 4
                         if name not in {"ready", "ack"} else 3 * 4)
               for name in ("src", "payload", "ready", "ack", "stale", "observed")]
src, payload, ready, ack, stale, observed = allocations
api = _hip()

kernel = tilelang.compile(bind_ipc(block_publication, ["src", "payload", "ready", "ack"]),
                          target="hip -mcpu=gfx936", execution_backend="ctypes")
for batch in range(BATCHES):
    seed = batch + 1
    # Distinct nonzero canaries make an out-of-range copy or scalar store
    # observable even when it would otherwise overwrite a zero-filled guard.
    for allocation, nbytes, fill in ((src, TOTAL * 4, CANARY_SRC_BYTE),
                                     (payload, TOTAL * 4, CANARY_PAYLOAD_BYTE),
                                     (ready, 12, 0), (ack, 12, 0),
                                     (stale, TOTAL * 4, CANARY_STALE_BYTE),
                                     (observed, OBSERVED_TOTAL * 4, CANARY_OBSERVED_BYTE)):
        _check(api.hipMemset(ctypes.c_void_p(allocation.data_ptr()), fill, nbytes), "batch reset")
    ctx.sync()
    kernel(src, payload, ready, ack, stale, observed, seed,
           src.device_view(), payload.device_view(), ready.device_view(), ack.device_view(), ctx.rank, ctx.world_size)
    event = ctypes.c_void_p()
    _check(api.hipEventCreate(ctypes.byref(event)), "hipEventCreate")
    _check(api.hipEventRecord(event, None), "hipEventRecord")
    deadline = time.monotonic() + 30.0
    while api.hipEventQuery(event) == 600:  # hipErrorNotReady
        if time.monotonic() > deadline:
            raise RuntimeError(f"rank={ctx.rank} batch={batch}: block publication timeout")
        time.sleep(0.01)
    _check(api.hipEventQuery(event), "hipEventQuery")
    api.hipEventDestroy(event)
    ctx.sync()

    host = (ctypes.c_uint32 * OBSERVED_TOTAL)()
    payload_host = (ctypes.c_uint32 * TOTAL)()
    src_host = (ctypes.c_uint32 * TOTAL)()
    stale_host = (ctypes.c_uint32 * TOTAL)()
    _copy(api, ctypes.addressof(host), observed.data_ptr(), OBSERVED_TOTAL * 4, 2)
    _copy(api, ctypes.addressof(payload_host), payload.data_ptr(), TOTAL * 4, 2)
    _copy(api, ctypes.addressof(src_host), src.data_ptr(), TOTAL * 4, 2)
    _copy(api, ctypes.addressof(stale_host), stale.data_ptr(), TOTAL * 4, 2)
    sources = ((ctx.rank - 1) % 4, (ctx.rank - 2) % 4, (ctx.rank - 3) % 4)
    edges = ((L0, sources[0], 0), (L1, sources[1], SLOT), (L2, sources[2], 2 * SLOT))
    for epoch in range(1, EPOCHS + 1):
        epoch_base = (epoch - 1) * TOTAL
        for edge, (length, sender, base) in enumerate(edges):
            for lane in range(length):
                expected = _payload_value(seed, epoch, sender, edge, lane)
                actual = host[epoch_base + base + lane]
                if actual != expected:
                    raise RuntimeError(
                        f"rank={ctx.rank} batch={batch} epoch={epoch} edge={edge} lane={lane}: "
                        f"{actual} != {expected}"
                    )
            for lane in range(length, SLOT):
                if host[epoch_base + base + lane] != CANARY_PAYLOAD:
                    raise RuntimeError(f"rank={ctx.rank} batch={batch} epoch={epoch} edge={edge}: slot tail corrupted")
        if any(host[epoch_base + 3 * SLOT + lane] != CANARY_OBSERVED for lane in range(GUARD)):
            raise RuntimeError(f"rank={ctx.rank} batch={batch} epoch={epoch}: observed guard corrupted")
    # Inspect the actual remote-copy allocation too: each slot tail and the
    # allocation guard must remain untouched, not merely the observed buffer.
    for edge, (length, _sender, base) in enumerate(edges):
        if any(payload_host[base + lane] != CANARY_PAYLOAD for lane in range(length, SLOT)):
            raise RuntimeError(f"rank={ctx.rank} batch={batch} edge={edge}: payload slot tail corrupted")
    if any(payload_host[3 * SLOT + lane] != CANARY_PAYLOAD for lane in range(GUARD)):
        raise RuntimeError(f"rank={ctx.rank} batch={batch}: payload guard corrupted")
    if any(src_host[3 * SLOT + lane] != CANARY_SRC for lane in range(GUARD)):
        raise RuntimeError(f"rank={ctx.rank} batch={batch}: source guard corrupted")
    if any(stale_host[3 * SLOT + lane] != CANARY_STALE for lane in range(GUARD)):
        raise RuntimeError(f"rank={ctx.rank} batch={batch}: stale guard corrupted")
for allocation in allocations:
    allocation.close()
ctx.close()
_check(mpi.MPI_Finalize(), "MPI_Finalize")
print(f"m2-block-litmus-ok rank={ctx.rank}", flush=True)
