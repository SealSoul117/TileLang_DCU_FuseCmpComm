#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;
using ffi::Array;
using ffi::String;
using ffi::Map;
using ffi::Any;

namespace {
constexpr const char* kRegisterPipelineStageAttr = "tl_register_pipeline_stage";
constexpr const char* kRegisterPipelineOrderAttr = "tl_register_pipeline_order";
constexpr const char* kRegisterPipelineAsyncStagesAttr =
    "tl_register_pipeline_async_stages";

inline bool IsScopeOrPrefix(const String& scope, const char* prefix) {
  std::string s = scope;
  std::string p = prefix;
  return s == p || (s.size() > p.size() && s.compare(0, p.size(), p) == 0 &&
                    s[p.size()] == '.');
}

inline bool IsLocalScope(const String& scope) {
  return IsScopeOrPrefix(scope, "local");
}

inline bool IsSharedScope(const String& scope) {
  return IsScopeOrPrefix(scope, "shared");
}

class RegisterPipelineClassifier : public StmtExprVisitor {
 public:
  static bool IsSharedToLocalCopy(const Stmt& stmt) {
    RegisterPipelineClassifier classifier;
    classifier(stmt);
    return classifier.has_local_store_ && classifier.reads_shared_;
  }

  static bool HasMmaCompute(const Stmt& stmt) {
    RegisterPipelineClassifier classifier;
    classifier(stmt);
    return classifier.has_mma_compute_;
  }

  static bool HasUnitExtentLoop(const Stmt& stmt) {
    RegisterPipelineClassifier classifier;
    classifier(stmt);
    return classifier.has_unit_extent_loop_;
  }

  static bool HasGlobalToLocalCopy(const Stmt& stmt) {
    RegisterPipelineClassifier classifier;
    classifier(stmt);
    return classifier.reads_global_;
  }

  static bool HasAnyLocalAccess(const Stmt& stmt) {
    RegisterPipelineClassifier classifier;
    classifier(stmt);
    return classifier.reads_local_ || classifier.has_local_store_;
  }

 private:
  void VisitStmt_(const ForNode* op) final {
    if (is_one(op->extent)) {
      has_unit_extent_loop_ = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    if (IsLocalScope(op->buffer.scope())) {
      has_local_store_ = true;
      bool old = in_local_store_value_;
      in_local_store_value_ = true;
      VisitExpr(op->value);
      in_local_store_value_ = old;
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    if (in_local_store_value_ && IsSharedScope(op->buffer.scope())) {
      reads_shared_ = true;
    } else if (op->buffer.scope() == "global") {
      reads_global_ = true;
    } else if (IsLocalScope(op->buffer.scope())) {
      reads_local_ = true;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallNode* op) final {
    if (auto* op_node = op->op.as<OpNode>()) {
      std::string op_name = op_node->name;
      if ((op_name == "tl.tvm_mmac")) {
            has_mma_compute_ = true;
      }
      StmtExprVisitor::VisitExpr_(op);
    }
  }

  bool in_local_store_value_ = false;
  bool has_local_store_ = false;
  bool reads_shared_ = false;
  bool reads_local_ = false;
  bool has_mma_compute_ = false;
  bool reads_global_ = false;
  bool has_unit_extent_loop_ = false;
};

class RegisterPipelinePlanner : public StmtExprMutator {
 public:
  Stmt VisitStmt_(const ForNode* op) final {
    For for_node = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    // Register pipeline is designed to refine an existing outer shared-memory
    // pipeline loop. Do not run on arbitrary inner loops.
    bool has_shared_pipeline_anno =
        op->annotations.count(tir::attr::software_pipeline_stage) &&
        op->annotations.count(tir::attr::software_pipeline_order);
    if (!has_shared_pipeline_anno) {
      return for_node;
    }

    int num_register_stages = 2;
    if (auto num_reg_stages_anno = op->annotations.Get("num_register_stages")) {
      if (const auto* imm = num_reg_stages_anno.value().as<IntImmNode>()) {
        num_register_stages = imm->value;
      }
    }
    if (num_register_stages <= 1) {
      return for_node;
    }

    if (for_node->kind != ForKind::kSerial) {
      return for_node;
    }
    const SeqStmtNode* seq = GetPipelineBodySeq(for_node->body);
    if (seq == nullptr) {
      return for_node;
    }

    std::vector<Stmt> components;
    components.reserve(seq->size());
    for (const Stmt& child : seq->seq) {
      bool has_unsupported_mma_loop = false;
      if (const auto* inner_seq =
              ExtractSplittableInnerSeq(child, &has_unsupported_mma_loop)) {
        for (const Stmt& inner : inner_seq->seq) {
          components.push_back(inner);
        }
      } else {
        // If MMA is wrapped by a loop with extent > 1, this pass cannot
        // safely infer register pipeline stages. Keep the original loop.
        if (has_unsupported_mma_loop) {
          return for_node;
        }
        components.push_back(child);
      }
    }

    const int n = static_cast<int>(components.size());
    if (n == 0) {
      return for_node;
    }

    std::vector<bool> is_shared_to_local(n, false);
    std::vector<bool> has_mma_compute(n, false);
    std::vector<bool> has_local_access(n, false);
    int first_register_producer_idx = -1;
    int first_compute_idx = -1;

    for (int i = 0; i < n; ++i) {
      const Stmt& s = components[i];
      if (RegisterPipelineClassifier::IsSharedToLocalCopy(s)) {
        is_shared_to_local[i] = true;
        if (first_register_producer_idx == -1) {
          first_register_producer_idx = i;
        }
      }
      if (RegisterPipelineClassifier::HasMmaCompute(s)) {
        has_mma_compute[i] = true;
        if (first_compute_idx == -1) {
          first_compute_idx = i;
        }
      }
      has_local_access[i] = RegisterPipelineClassifier::HasAnyLocalAccess(s);
    }
    if (first_register_producer_idx == -1 || first_compute_idx == -1 ||
        first_register_producer_idx >= first_compute_idx) {
      return for_node;
    }
    int compute_stage = 1;
    if (auto stage_anno = op->annotations.Get(kRegisterPipelineStageAttr)) {
      if (auto old_stages = stage_anno.value().try_cast<Array<Integer>>()) {
        for (const Integer& stage : old_stages.value()) {
          compute_stage = std::max(compute_stage, static_cast<int>(stage->value));
        }
      }
    } else if (auto stage_anno =
                   op->annotations.Get(tir::attr::software_pipeline_stage)) {
      if (auto old_stages = stage_anno.value().try_cast<Array<Integer>>()) {
        for (const Integer& stage : old_stages.value()) {
          compute_stage = std::max(compute_stage, static_cast<int>(stage->value));
        }
      }
    }
    int register_stage = std::max(0, compute_stage - 1);

    std::vector<Integer> orders(n, Integer(-1));
    std::vector<Integer> stages(n, Integer(compute_stage));
    if (auto order_anno = op->annotations.Get(kRegisterPipelineOrderAttr)) {
      if (auto old_orders = order_anno.value().try_cast<Array<Integer>>()) {
        if (old_orders.value().size() == components.size()) {
          for (int i = 0; i < n; ++i) {
            orders[i] = old_orders.value()[i];
          }
        }
      }
    } else if (auto order_anno =
                   op->annotations.Get(tir::attr::software_pipeline_order)) {
      if (auto old_orders = order_anno.value().try_cast<Array<Integer>>()) {
        if (old_orders.value().size() == components.size()) {
          for (int i = 0; i < n; ++i) {
            orders[i] = old_orders.value()[i];
          }
        }
      }
    }

    for (int i = 0; i < n; ++i) {
      if (orders[i]->value == -1) {
        orders[i] = Integer(i);
      }
      if (i < first_register_producer_idx) {
        stages[i] = Integer(0);
        continue;
      }

      if (i < first_compute_idx) {
        stages[i] = Integer(register_stage);
        continue;
      }

      if (has_mma_compute[i]) {
        stages[i] = Integer(compute_stage);
      } else if (is_shared_to_local[i]) {
        stages[i] = Integer(register_stage);
      } else if (has_local_access[i] && i < first_compute_idx) {
        stages[i] = Integer(register_stage);
      } else {
        stages[i] = Integer(compute_stage);
      }
    }

    Map<String, Any> annotations;
    for (const auto& kv : for_node->annotations) {
      const String& key = kv.first;
      // Keep num_register_stages so InjectRegisterSoftwarePipeline can size
      // register ping-pong banks consistently with this pass.
      if (key != kRegisterPipelineStageAttr && key != kRegisterPipelineOrderAttr &&
          key != kRegisterPipelineAsyncStagesAttr) {
        annotations.Set(key, kv.second);
      }
    }
    annotations.Set(kRegisterPipelineStageAttr, Array<Integer>(stages));
    annotations.Set(kRegisterPipelineOrderAttr, Array<Integer>(orders));
    if (auto async_stages = op->annotations.Get(kRegisterPipelineAsyncStagesAttr)) {
      annotations.Set(kRegisterPipelineAsyncStagesAttr, async_stages.value());
    } else if (auto sw_async = op->annotations.Get(
                   tir::attr::software_pipeline_async_stages)) {
      // InjectRegisterSoftwarePipeline only consults tl_register_pipeline_async_stages
      // when wrapping async producers in async_scope. Without this, global→shared
      // copies stay outside async_scope and passes such as LowerSharedGlobalCopy
      // (which require in_async) never match.
      annotations.Set(kRegisterPipelineAsyncStagesAttr, sw_async.value());
    }
    return For(for_node->loop_var, for_node->min, for_node->extent,
               for_node->kind, for_node->body, for_node->thread_binding,
               std::move(annotations));
  }

 private:
  const SeqStmtNode* ExtractSplittableInnerSeq(
      const Stmt& stmt, bool* has_unsupported_mma_loop) const {
    const auto* br = stmt.as<BlockRealizeNode>();
    if (!br || !is_one(br->predicate)) {
      return nullptr;
    }
    if (!RegisterPipelineClassifier::HasMmaCompute(br->block->body)) {
      return nullptr;
    }
    Stmt current = br->block->body;
    while (true) {
      if (const auto* seq = current.as<SeqStmtNode>()) {
        return seq;
      }
      if (const auto* inner_br = current.as<BlockRealizeNode>()) {
        current = inner_br->block->body;
        continue;
      }
      if (const auto* attr = current.as<AttrStmtNode>()) {
        current = attr->body;
        continue;
      }
      if (const auto* let_stmt = current.as<LetStmtNode>()) {
        current = let_stmt->body;
        continue;
      }
      if (const auto* for_stmt = current.as<ForNode>()) {
        if (is_one(for_stmt->extent)) {
          current = for_stmt->body;
          continue;
        }
        if (has_unsupported_mma_loop != nullptr &&
            RegisterPipelineClassifier::HasMmaCompute(for_stmt->body)) {
          *has_unsupported_mma_loop = true;
        }
        return nullptr;
      }
      if (const auto* if_then_else = current.as<IfThenElseNode>()) {
        if (!if_then_else->else_case.defined()) {
          current = if_then_else->then_case;
          continue;
        }
      }
      return nullptr;
    }
  }

  const SeqStmtNode* GetPipelineBodySeq(const Stmt& stmt) const {
    Stmt current = stmt;
    while (true) {
      if (const auto* seq = current.as<SeqStmtNode>()) {
        return seq;
      }
      if (const auto* br = current.as<BlockRealizeNode>()) {
        current = br->block->body;
        continue;
      }
      if (const auto* attr = current.as<AttrStmtNode>()) {
        current = attr->body;
        continue;
      }
      if (const auto* let_stmt = current.as<LetStmtNode>()) {
        current = let_stmt->body;
        continue;
      }
      if (const auto* allocate = current.as<AllocateNode>()) {
        current = allocate->body;
        continue;
      }
      if (const auto* decl_buffer = current.as<DeclBufferNode>()) {
        current = decl_buffer->body;
        continue;
      }
      if (const auto* if_then_else = current.as<IfThenElseNode>()) {
        if (!if_then_else->else_case.defined()) {
          current = if_then_else->then_case;
          continue;
        }
      }
      return nullptr;
    }
  }
};

}  // namespace

tir::transform::Pass RegisterPipelinePlanning() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule&, const PassContext&) {
    auto* fptr = f.CopyOnWrite();
    fptr->body = RegisterPipelinePlanner()(fptr->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.RegisterPipelinePlanning", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.RegisterPipelinePlanning",
                        RegisterPipelinePlanning);
}

}  // namespace tl
}  // namespace tvm
