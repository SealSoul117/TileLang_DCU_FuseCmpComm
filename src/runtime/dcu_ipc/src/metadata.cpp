#include "metadata.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace dcu_ipc::metadata {
namespace {
Status invalid(const char* detail) {
  return Status::error(StatusCode::InvalidArgument, "metadata", detail);
}
bool valid_kind(OpKind kind) {
  return kind == OpKind::Init || kind == OpKind::Alloc || kind == OpKind::Sync ||
         kind == OpKind::Free || kind == OpKind::Finalize;
}
bool valid_memory(MemoryKind kind) {
  return kind == MemoryKind::DeviceDefault || kind == MemoryKind::FineGrained;
}
bool tagged(OpKind kind) { return kind == OpKind::Alloc || kind == OpKind::Free; }
}

Status validate_init_options(const InitOptions& options) {
  if (options.selection == DeviceSelection::Auto && options.ordinal == -1)
    return Status::success();
  if (options.selection == DeviceSelection::ExplicitOrdinal && options.ordinal >= 0)
    return Status::success();
  return invalid("auto needs ordinal=-1; explicit ordinal must be nonnegative");
}

Status validate_tag(const char* tag, const std::vector<std::string>& live_tags) {
  if (!tag) return invalid("tag is null");
  // Inspect at most max+1 characters, so overlong tags are rejected, never truncated.
  std::size_t length = 0;
  while (length <= max_tag_bytes && tag[length] != '\0') ++length;
  if (length == 0 || length > max_tag_bytes) return invalid("tag empty or longer than 63 bytes");
  if (std::find(live_tags.begin(), live_tags.end(), std::string(tag, length)) != live_tags.end())
    return Status::error(StatusCode::InvalidState, "metadata", "tag already live");
  return Status::success();
}

Status validate_peer(int peer, int npes) {
  if (npes < 1 || peer < 0 || peer >= npes) return invalid("peer outside communicator");
  return Status::success();
}

Status validate_range(std::size_t nbytes, std::size_t byte_offset, std::size_t access_bytes) {
  if (access_bytes == 0 || byte_offset > nbytes || access_bytes > nbytes - byte_offset)
    return invalid("access outside allocation or zero length");
  return Status::success();
}

Status validate_typed_range(std::size_t nbytes, std::size_t element_index,
                            std::size_t element_count, std::size_t element_size,
                            std::size_t alignment) {
  // alignment is alignof(T), hence a nonzero power of two.
  if (!element_size || !alignment || (alignment & (alignment - 1)) || element_count == 0)
    return invalid("invalid typed width/count/alignment");
  constexpr auto limit = std::numeric_limits<std::size_t>::max();
  if (element_index > limit / element_size || element_count > limit / element_size)
    return invalid("typed byte multiplication overflow");
  const std::size_t offset = element_index * element_size;
  const std::size_t length = element_count * element_size;
  if (offset % alignment) return invalid("typed byte offset not naturally aligned");
  return validate_range(nbytes, offset, length);
}

Status make_header(OpKind kind, std::uint64_t sequence, std::uint64_t allocation_id,
                   std::uint64_t generation, const char* tag, std::size_t nbytes,
                   MemoryKind memory_kind, OperationHeader* out) {
  if (!out || !valid_kind(kind) || !valid_memory(memory_kind)) return invalid("header argument or enum invalid");
  if (nbytes > std::numeric_limits<std::uint64_t>::max()) return invalid("nbytes exceeds wire width");
  if (tagged(kind)) {
    auto status = validate_tag(tag, {});
    if (!status.ok()) return status;
    if (nbytes == 0) return invalid("tagged allocation has zero bytes");
  } else if ((tag && tag[0] != '\0') || nbytes != 0) {
    return invalid("untagged operation must have empty tag and zero bytes");
  }
  OperationHeader candidate{};
  candidate.version = protocol_version;
  candidate.op_kind = static_cast<std::uint32_t>(kind);
  candidate.memory_kind = static_cast<std::uint32_t>(memory_kind);
  candidate.sequence = sequence;
  candidate.allocation_id = allocation_id;
  candidate.generation = generation;
  candidate.nbytes = static_cast<std::uint64_t>(nbytes);
  if (tagged(kind)) {
    candidate.tag_length = static_cast<std::uint32_t>(std::strlen(tag));
    std::memcpy(candidate.tag, tag, candidate.tag_length);
  }
  *out = candidate;
  return Status::success();
}

Status validate_header(const OperationHeader& header) {
  if (header.version != protocol_version)
    return Status::error(StatusCode::SymmetryMismatch, "header", "protocol version mismatch");
  if (header.op_kind < std::uint32_t(OpKind::Init) ||
      header.op_kind > std::uint32_t(OpKind::Finalize) ||
      header.memory_kind > std::uint32_t(MemoryKind::FineGrained))
    return invalid("wire operation or memory kind invalid");
  const auto kind = static_cast<OpKind>(header.op_kind);
  if (header.tag_length > max_tag_bytes || header.tag[header.tag_length] != '\0')
    return invalid("wire tag length/termination invalid");
  for (std::size_t i = 0; i < header.tag_length; ++i)
    if (header.tag[i] == '\0') return invalid("wire tag contains embedded NUL");
  for (std::size_t i = header.tag_length + 1; i < sizeof(header.tag); ++i)
    if (header.tag[i] != '\0') return invalid("wire tag padding not zero");
  if (tagged(kind)) {
    if (header.tag_length == 0 || header.nbytes == 0) return invalid("tagged operation missing tag or size");
  } else if (header.tag_length != 0 || header.nbytes != 0) {
    return invalid("untagged operation carries tag or bytes");
  }
  return Status::success();
}

Status validate_collective_headers(const std::vector<OperationHeader>& headers) {
  if (headers.empty()) return invalid("empty collective header set");
  for (std::size_t rank = 0; rank < headers.size(); ++rank) {
    auto status = validate_header(headers[rank]);
    if (!status.ok()) {
      status.detail += " at rank " + std::to_string(rank);
      return status;
    }
  }
  const auto& first = headers.front();
  for (std::size_t rank = 1; rank < headers.size(); ++rank) {
    const auto& h = headers[rank];
    if (h.version != first.version || h.sequence != first.sequence ||
        h.op_kind != first.op_kind || h.memory_kind != first.memory_kind ||
        h.allocation_id != first.allocation_id || h.generation != first.generation ||
        h.nbytes != first.nbytes || h.tag_length != first.tag_length ||
        std::memcmp(h.tag, first.tag, sizeof(h.tag)) != 0)
      return Status::error(StatusCode::SymmetryMismatch, "collective header",
                           "rank " + std::to_string(rank) + " differs from rank 0");
  }
  return Status::success();
}

} // namespace dcu_ipc::metadata
