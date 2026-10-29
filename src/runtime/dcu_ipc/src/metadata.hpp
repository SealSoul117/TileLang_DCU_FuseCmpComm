#pragma once

#include <dcu_ipc/types.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace dcu_ipc::metadata {

inline constexpr std::uint32_t protocol_version = 1;
inline constexpr std::size_t max_tag_bytes = 63;

enum class OpKind : std::uint8_t { Init = 1, Alloc = 2, Sync = 3, Free = 4, Finalize = 5 };

// Fixed-size host wire record. Zero-initialize before setting every field;
// no host pointer, variable-length payload, or C++ object crosses MPI.
struct OperationHeader {
  std::uint32_t version;
  std::uint32_t op_kind;
  std::uint32_t memory_kind;
  std::uint32_t tag_length;
  std::uint64_t sequence;
  std::uint64_t allocation_id;
  std::uint64_t generation;
  std::uint64_t nbytes;
  char tag[max_tag_bytes + 1];
};
static_assert(std::is_standard_layout<OperationHeader>::value, "wire header layout");
static_assert(std::is_trivially_copyable<OperationHeader>::value, "wire header copyability");

Status validate_init_options(const InitOptions& options);
Status validate_tag(const char* tag, const std::vector<std::string>& live_tags);
Status validate_peer(int peer, int npes);
Status validate_range(std::size_t nbytes, std::size_t byte_offset, std::size_t access_bytes);
Status validate_typed_range(std::size_t nbytes, std::size_t element_index,
                            std::size_t element_count, std::size_t element_size,
                            std::size_t alignment);
Status make_header(OpKind kind, std::uint64_t sequence, std::uint64_t allocation_id,
                   std::uint64_t generation, const char* tag, std::size_t nbytes,
                   MemoryKind memory_kind, OperationHeader* out);
Status validate_header(const OperationHeader& header);
Status validate_collective_headers(const std::vector<OperationHeader>& headers);

} // namespace dcu_ipc::metadata
