"""Pure host-side ABI mapping for the external matched HIP diagnostic."""


def diagnostic_hip_arguments(a, b, partial, ready, out, rank: int, world_size: int) -> tuple[int, ...]:
    """Return the exported kernel's nine arguments in source-kernel order.

    The generated matched source takes IPC descriptor tables before the local
    payload pointers: ``a, b, partial_view, ready_view, out, partial,
    ready, rank, world_size``.  Keeping this standalone makes the mapping
    testable without importing TileLang or loading HIP runtime libraries.
    """
    return (
        a.data_ptr(), b.data_ptr(), partial.device_view().data_ptr(), ready.device_view().data_ptr(),
        out.data_ptr(), partial.data_ptr(), ready.data_ptr(), rank, world_size,
    )
