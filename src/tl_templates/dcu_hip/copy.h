  #pragma once

  #include "common.h"

  using f32 = float;
  // using f16 = _Float16;

  using u8 = std::uint8_t;
  using u16 = std::uint16_t;
  using u32 = std::uint32_t;

  using index_t = u32;

  using ck_tile::int32x4_t;

  struct __attribute__((packed)) buffer_resource {
    const void *ptr;
    uint32_t range;
    uint32_t config;
  };
  # define CK_TILE_BUFFER_RESOURCE_3RD_DWORD 0x00020000

  __device__ void init_m0(uint32_t m0_value) {
    asm volatile("s_mov_b32 m0, %0" : : "s"(m0_value) : "memory");
  }

  __device__ void inc_m0(uint32_t m0_inc) {
    asm volatile("s_add_u32 m0, %0, m0" : : "n"(m0_inc) : "memory");
  }

  #define UPDATE_WAVE_BUFFER_RESOURCE(res, stride)                       \
      do {                                                               \
          /* 1. 提取 64 位基地址，确保低位不进行符号位扩展 */               \
          uint64_t __current_addr = (static_cast<uint64_t>((res).y) << 32) | \
                                    (static_cast<uint32_t>((res).x));    \
                                                                        \
          /* 2. 增加步长 (自动处理类型提升) */                             \
          __current_addr += (stride);                                    \
                                                                        \
          /* 3. 写回分量到 SGPRs */                                       \
          (res).x = static_cast<int32_t>(__current_addr);                \
          (res).y = static_cast<int32_t>(__current_addr >> 32);          \
      } while (0)

  namespace tl {

  // AMDGPU automatically commit memory fence
  TL_DEVICE void cp_async_commit() {}

  // Global Memory only fence
  __device__ void async_gld_fence(index_t cnt) {
    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
  }

  // Global Memory and Shared Memory fence
  __device__ void async_gld_sld_fence(index_t cnt) {
    asm volatile("s_waitcnt lgkmcnt(%0)" : : "n"(cnt) : "memory");
  }

  __device__ void wave_barrier() { asm volatile("s_barrier" : : : "memory"); }

  template <int N = 0> TL_DEVICE void cp_async_wait() {
    async_gld_fence(N);
    // or
    // async_gld_sld_fence(N);
  }

  template <bool pre_nop = false>
  CK_TILE_DEVICE void async_buffer_load_dword_v(void *smem, int32x4_t rsrc,
                                                index_t voffset) {
    auto const lds_ptr_sgpr =
        __builtin_amdgcn_readfirstlane((reinterpret_cast<uintptr_t>(smem)));
    asm volatile("s_mov_b32 m0, %0; \n\t"
                "buffer_load_dword %1, %2, 0 offen lds;\n\t" ::"s"(lds_ptr_sgpr),
                "v"(voffset), "s"(rsrc)
                : "memory");
  }

  template <int smem_offset =0, bool pre_nop = false>
  CK_TILE_DEVICE void async_buffer_load_dwordx4_v(void *smem, int32x4_t rsrc,
                                                index_t voffset) {
    auto const lds_ptr_sgpr =
        __builtin_amdgcn_readfirstlane((reinterpret_cast<uintptr_t>(smem)));
    asm volatile("s_add_u32 m0, %0, %3 \n\t"
                "buffer_load_dwordx4 %1, %2, 0, offen offset:0, lds\n\t" ::"s"(lds_ptr_sgpr),
                "v"(voffset), "s"(rsrc), "n"(smem_offset)
                : "memory");
  }

  template <int N>
  TL_DEVICE void cp_async_gs(void *lds_base_ptr, int32x4_t res, int offset) {
    if constexpr (N == 16) {
      async_buffer_load_dwordx4_v(
          lds_base_ptr,
          res,
          offset
        );
    }
  }

  template <int N, int smem_offset = 0>
  TL_DEVICE void cp_async_gs_idxen(void *lds_base_ptr, int32x4_t res, int offset, int idxen) {
      typedef uint32_t uint32x2_t __attribute__((ext_vector_type(2)));
      uint32x2_t v_addr = {idxen, offset};
      auto const lds_ptr_sgpr =
        __builtin_amdgcn_readfirstlane((reinterpret_cast<uintptr_t>(lds_base_ptr)));
      
    asm volatile("s_add_u32 m0, %0, %3 \n\t"
                "buffer_load_dwordx4 %1, %2, 0, idxen offen offset:0, lds\n\t" ::"s"(lds_ptr_sgpr),
                "v"(v_addr), "s"(res), "n"(smem_offset)
                : "memory");
  }

  template <uint32_t StructBit = 0>
  TL_DEVICE int32x4_t make_wave_buffer_resource(const void* ptr,
                            uint32_t size = 0xffffffff)
  {
      buffer_resource res{
          ptr,
          size,
          CK_TILE_BUFFER_RESOURCE_3RD_DWORD
      };

      int32x4_t r = __builtin_bit_cast(int32x4_t, res);

      if constexpr (StructBit != 0) {
          static_assert(StructBit <= 15,
                        "StructBit must fit in 4 bits for GCN structured buffer");

          constexpr uint32_t IS_STRUCTURED = 0x40000000u;
          constexpr uint32_t STRUCT_SCALE  = 1u << (16 + StructBit);

          uint32_t addr_hi = static_cast<uint32_t>(r.y) & 0x0000FFFFu;
          r.y = static_cast<int32_t>(
              addr_hi | IS_STRUCTURED | STRUCT_SCALE);
      }

      r.x = __builtin_amdgcn_readfirstlane(r.x);
      r.y = __builtin_amdgcn_readfirstlane(r.y);
      r.z = __builtin_amdgcn_readfirstlane(r.z);
      r.w = __builtin_amdgcn_readfirstlane(r.w);

      return r;
  }

  TL_DEVICE void ds_read_m32x16_b16(float32x4& dst, uint32_t lds_base_ptr) {
    asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t"
                : "+v"(dst)
                : "v"(lds_base_ptr));
  }

  TL_DEVICE void ds_read_m32x32_b8(float32x4& dst, uint32_t lds_base_ptr) {
    asm volatile("ds_read_m32x32_b8 %0, %1 offset:0\n\t"
                : "=v"(dst)
                : "v"(lds_base_ptr)
                : "memory");
  }


  TL_DEVICE void ds_read_b64(
      half_t* dst_base, 
      uint32_t lds_byte_addr) 
  {
      uint2* dst_ptr = (uint2*)(dst_base);

      asm volatile(
          "ds_read_b64 %0, %1 offset:0\n\t"
          : "=v"(*dst_ptr)       // %0: 写入的本地寄存器 (64位)
          : "v"(lds_byte_addr)   // %1: 算好的共享内存字节地址 (32位 VGPR)
          : "memory"
      );
  }
  template <int offset>
  TL_DEVICE void ds_read_b64_N(
      half_t* dst_base, 
      uint32_t lds_byte_addr) 
  {
      uint2* dst_ptr = (uint2*)(dst_base);

      asm volatile(
          "ds_read_b64 %0, %1 offset:%2\n\t"
          : "=v"(*dst_ptr)       // %0: 写入的本地寄存器 (64位)
          : "v"(lds_byte_addr), "n"(offset)   // %1: 算好的共享内存字节地址 (32位 VGPR)
          : "memory"
      );
  }
  template <int offset>
  TL_DEVICE void ds_read_vector_N(float32x4& dst, uint32_t  lds_base_ptr)
  {
      asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2\n\t"
                  : "+v"(dst)
                  : "v"(lds_base_ptr), "n"(offset));
  }

  template <int N>
  TL_DEVICE void cp_async_gs_conditional(void *lds_base_ptr,
                                        void *global_base_ptr, bool cond) {
    if constexpr (N == 16) {
      *(uint4 *)lds_base_ptr =
          cond ? *(uint4 *)global_base_ptr : make_uint4(0, 0, 0, 0);
    } else if constexpr (N == 8) {
      *(uint2 *)lds_base_ptr =
          cond ? *(uint2 *)global_base_ptr : make_uint2(0, 0);
    } else {
      if (cond) {
        async_buffer_load_dword_v(
            lds_base_ptr,
            make_wave_buffer_resource(((int32_t *)global_base_ptr) - threadIdx.x),
            threadIdx.x * N /*assume 4 bytes*/);
      } else {
        *(uint4 *)lds_base_ptr = make_uint4(0, 0, 0, 0);
      }
    }
  }



  } // namespace tl
