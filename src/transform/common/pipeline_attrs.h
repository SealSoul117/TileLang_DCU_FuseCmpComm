/*!
 * \file pipeline_attrs.h
 * \brief For-loop annotation keys for T.Pipelined software / register pipeline.
 */
#ifndef TVM_TL_TRANSFORM_COMMON_PIPELINE_ATTRS_H_
#define TVM_TL_TRANSFORM_COMMON_PIPELINE_ATTRS_H_

#include <algorithm>
#include <tvm/tir/stmt.h>

namespace tvm {
namespace tl {

static constexpr const char *kEnableRegisterPipeline = "enable_register_pipeline";
static constexpr const char *kNumRegisterStages = "num_register_stages";
static constexpr const char *kEnableWarpDivergence = "enable_warp_divergence";
static constexpr const char *kRegisterPipelineApplied = "tl_register_pipeline_applied";
static constexpr int kDefaultNumRegisterStages = 2;

inline bool LoopAnnotationIsTruthy(const tir::ForNode *loop, const char *key) {
  if (loop == nullptr) {
    return false;
  }
  if (auto val = loop->annotations.Get(key)) {
    if (const auto *imm = val.value().as<IntImmNode>()) {
      return imm->value != 0;
    }
  }
  return false;
}

inline int LoopAnnotationInt(const tir::ForNode *loop, const char *key, int default_value = 0) {
  if (loop == nullptr) {
    return default_value;
  }
  if (auto val = loop->annotations.Get(key)) {
    if (const auto *imm = val.value().as<IntImmNode>()) {
      return static_cast<int>(imm->value);
    }
  }
  return default_value;
}

inline bool LoopHasRegisterPipeline(const tir::ForNode *loop) {
  return LoopAnnotationIsTruthy(loop, kRegisterPipelineApplied) ||
         LoopAnnotationIsTruthy(loop, kEnableRegisterPipeline) ||
         LoopAnnotationInt(loop, kNumRegisterStages) > 1;
}

inline bool LoopHasWarpDivergence(const tir::ForNode *loop) {
  return LoopAnnotationIsTruthy(loop, kEnableWarpDivergence);
}

inline int LoopNumStages(const tir::ForNode *loop) {
  return LoopAnnotationInt(loop, "num_stages", 0);
}

/*! \brief Pipeline depth from num_stages, post-inject annotation, or max(stage)+1. */
inline int GetPipelineNumStages(const tir::ForNode *loop) {
  if (loop == nullptr) {
    return 0;
  }
  int ns = LoopAnnotationInt(loop, "num_stages", 0);
  if (ns > 0) {
    return ns;
  }
  ns = LoopAnnotationInt(loop, "tl_pipelined_num_stages", 0);
  if (ns > 0) {
    return ns;
  }
  if (auto stages_anno = loop->annotations.Get(tir::attr::software_pipeline_stage)) {
    auto stages = Downcast<ffi::Array<Integer>>(stages_anno.value());
    int max_stage = -1;
    for (const auto &stage : stages) {
      max_stage = std::max(max_stage, static_cast<int>(stage->value));
    }
    if (max_stage >= 0) {
      return max_stage + 1;
    }
  }
  return 0;
}

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_PIPELINE_ATTRS_H_
