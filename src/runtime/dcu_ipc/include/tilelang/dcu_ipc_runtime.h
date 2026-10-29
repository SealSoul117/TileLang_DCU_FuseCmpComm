#pragma once

// Narrow stable C ABI for ctypes and standalone TileLang runners.  MPI must
// already be initialized by the caller with MPI_THREAD_FUNNELED or stronger.
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct tl_dcu_ipc_context tl_dcu_ipc_context;
typedef struct tl_dcu_ipc_allocation tl_dcu_ipc_allocation;
int tl_dcu_ipc_init_world(int device_ordinal, tl_dcu_ipc_context** out);
int tl_dcu_ipc_rank(const tl_dcu_ipc_context* context);
int tl_dcu_ipc_world_size(const tl_dcu_ipc_context* context);
int tl_dcu_ipc_alloc(tl_dcu_ipc_context* context, const char* tag, size_t nbytes,
                     int fine_grained, tl_dcu_ipc_allocation** out);
void* tl_dcu_ipc_local_ptr(const tl_dcu_ipc_allocation* allocation);
int tl_dcu_ipc_device_view_words(tl_dcu_ipc_context* context,
                                 const tl_dcu_ipc_allocation* allocation,
                                 uint64_t** out_device_words);
int tl_dcu_ipc_sync(tl_dcu_ipc_context* context);
int tl_dcu_ipc_free(tl_dcu_ipc_context* context, tl_dcu_ipc_allocation** allocation);
int tl_dcu_ipc_finalize(tl_dcu_ipc_context** context);
const char* tl_dcu_ipc_last_error(void);
#ifdef __cplusplus
}
#endif
