/*!
 * \file inject_dcu_async_copy_idxen.cc
 * \brief Convert A-matrix tl.dcu_async_copy to tl.dcu_async_copy_idxen with structured
 *        buffer resource for DCU global-to-shared loads.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "../op/builtin.h"
#include "inject_utils.h"

using tvm::ffi::GetRef;

namespace tvm {
namespace tl {

using namespace tir;

namespace {

const Op& DCUAsyncCopyOp() { return Op::Get("tl.dcu_async_copy"); }
const Op& DCUAsyncCopyIdxenOp() { return Op::Get("tl.dcu_async_copy_idxen"); }
const Op& MakeDCUResourceOp() { return Op::Get("tl.make_dcu_resource"); }

bool IsABufferName(const std::string& name) {
  return name == "A" || name == "A_1";
}

std::string GetGlobalBufferName(const PrimExpr& expr) {
  if (const auto* var = expr.as<VarNode>()) {
    return var->name_hint;
  }
  if (const auto* load = expr.as<BufferLoadNode>()) {
    return load->buffer->name;
  }
  return "";
}

// Maximum pattern width (tile_width): 2^13 = 8192 bytes.
constexpr int64_t kMaxTileWidth = 1LL << 13;

int64_t LargestPow2LessThan(int64_t k) {
  ICHECK_GT(k, 1);
  int64_t p = 1;
  while ((p << 1) < k) {
    p <<= 1;
  }
  return p;
}

int64_t TileWidthFromK(int64_t k) {
  int64_t tile_width = LargestPow2LessThan(k);
  if (tile_width > kMaxTileWidth) {
    tile_width = kMaxTileWidth;
  }
  return tile_width;
}

int StructBitFromTileWidth(int64_t tile_width) {
  int bits = 0;
  while ((1LL << bits) < tile_width) {
    ++bits;
  }
  ICHECK_EQ(1LL << bits, tile_width);
  return bits;
}

struct ADimInfo {
  std::optional<int64_t> k_dim;
  int elem_bytes{1};
};

class KDimCollector : public StmtExprVisitor {
 public:
  ADimInfo info;

 private:
  void VisitStmt_(const DeclBufferNode* op) final {
    if (IsABufferName(op->buffer->name)) {
      info.elem_bytes = op->buffer->dtype.bytes();
      if (op->buffer->shape.size() == 2) {
        if (const auto* imm = op->buffer->shape[1].as<IntImmNode>()) {
          info.k_dim = imm->value;
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

std::optional<ADimInfo> ExtractADimInfoFromModule(const IRModule& mod) {
  for (const auto& kv : mod->functions) {
    if (const auto* func = kv.second.as<PrimFuncNode>()) {
      KDimCollector collector;
      collector(func->body);
      if (collector.info.k_dim.has_value()) {
        return collector.info;
      }
    }
  }
  return std::nullopt;
}

class InfoCollector : public StmtExprVisitor {
 public:
  std::unordered_set<const VarNode*> a_dcu_res_vars;

 private:
  void VisitStmt_(const LetStmtNode* op) final {
    if (const auto* call = op->value.as<CallNode>()) {
      if (call->op.same_as(MakeDCUResourceOp()) && call->args.size() >= 1) {
        if (IsABufferName(GetGlobalBufferName(call->args[0]))) {
          a_dcu_res_vars.insert(op->var.get());
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

PrimExpr StripCast(const PrimExpr& expr) {
  if (const auto* cast = expr.as<CastNode>()) {
    return cast->value;
  }
  return expr;
}

void CollectAddTerms(const PrimExpr& expr, std::vector<PrimExpr>* terms) {
  PrimExpr e = StripCast(expr);
  if (const auto* add = e.as<AddNode>()) {
    CollectAddTerms(add->a, terms);
    CollectAddTerms(add->b, terms);
  } else if (const auto* sub = e.as<SubNode>()) {
    CollectAddTerms(sub->a, terms);
    terms->push_back(-sub->b);
  } else {
    terms->push_back(e);
  }
}

// If `term` is (expr * coeff) or (coeff * expr) with coeff divisible by K, return expr * (coeff/K).
std::optional<PrimExpr> TryExtractIdxenTerm(const PrimExpr& term, int64_t k_dim) {
  PrimExpr e = StripCast(term);
  const auto* mul = e.as<MulNode>();
  if (mul == nullptr) {
    if (const auto* imm = e.as<IntImmNode>()) {
      if (imm->value % k_dim == 0) {
        int64_t quotient = imm->value / k_dim;
        if (quotient == 0) {
          return make_zero(e.dtype());
        }
        return make_const(e.dtype(), quotient);
      }
    }
    return std::nullopt;
  }

  auto try_imm = [&](const PrimExpr& other, int64_t coeff) -> std::optional<PrimExpr> {
    if (coeff % k_dim != 0) {
      return std::nullopt;
    }
    int64_t quotient = coeff / k_dim;
    if (quotient == 1) {
      return other;
    }
    if (quotient == 0) {
      return make_zero(other.dtype());
    }
    return other * make_const(other.dtype(), quotient);
  };

  if (const auto* imm = mul->a.as<IntImmNode>()) {
    return try_imm(mul->b, imm->value);
  }
  if (const auto* imm = mul->b.as<IntImmNode>()) {
    return try_imm(mul->a, imm->value);
  }
  return std::nullopt;
}

class IdxenInjector : public StmtExprMutator {
 public:
  IdxenInjector(int64_t k_dim, int64_t tile_width, int struct_bit,
                std::unordered_set<const VarNode*> a_dcu_res_vars)
      : k_dim_(k_dim),
        tile_width_(tile_width),
        struct_bit_(struct_bit),
        a_dcu_res_vars_(std::move(a_dcu_res_vars)) {}

  static PrimFunc Substitute(PrimFunc func, int64_t k_dim, int64_t tile_width, int struct_bit,
                             const std::unordered_set<const VarNode*>& a_dcu_res_vars) {
    IdxenInjector mutator(k_dim, tile_width, struct_bit, a_dcu_res_vars);
    auto* n = func.CopyOnWrite();
    n->body = mutator(std::move(n->body));
    return func;
  }

 private:
  bool IsADcuRes(const PrimExpr& expr) const {
    if (const auto* var = expr.as<VarNode>()) {
      return a_dcu_res_vars_.count(var) > 0;
    }
    return false;
  }

  // Split offset = sum(K-scaled terms) + sum(rest) into:
  //   idxen = sum(term / K) for terms whose coefficient is a multiple of K
  //   new_offset = idxen * (K - tile_width) + sum(rest)
  std::pair<PrimExpr, PrimExpr> SplitSrcOffset(const PrimExpr& offset) {
    auto split_scalar = [&](const PrimExpr& base) -> std::pair<PrimExpr, PrimExpr> {
      std::vector<PrimExpr> terms;
      CollectAddTerms(base, &terms);

      PrimExpr idxen = make_zero(base.dtype());
      PrimExpr offset_rest = make_zero(base.dtype());
      for (const PrimExpr& term : terms) {
        if (auto idxen_term = TryExtractIdxenTerm(term, k_dim_)) {
          idxen = idxen + idxen_term.value();
        } else {
          offset_rest = offset_rest + term;
        }
      }

      idxen = analyzer_.Simplify(idxen);
      offset_rest = analyzer_.Simplify(offset_rest);
      const int64_t k_residual = k_dim_ - tile_width_;
      PrimExpr new_offset =
          analyzer_.Simplify(idxen * make_const(base.dtype(), k_residual) + offset_rest);
      return {new_offset, idxen};
    };

    if (const auto* ramp = offset.as<RampNode>()) {
      auto [new_base, idxen] = split_scalar(ramp->base);
      PrimExpr new_off = Ramp(new_base, ramp->stride, ramp->lanes);
      return {new_off, idxen};
    }
    return split_scalar(offset);
  }

  PrimExpr RewriteMakeDCUResource(const CallNode* call) {
    if (call->args.size() >= 3) {
      return GetRef<PrimExpr>(call);
    }
    if (!IsABufferName(GetGlobalBufferName(call->args[0]))) {
      return GetRef<PrimExpr>(call);
    }
    Array<PrimExpr> new_args = call->args;
    new_args.push_back(IntImm(DataType::Int(32), struct_bit_));
    return Call(call->dtype, call->op, new_args);
  }

  PrimExpr RewriteDCUAsyncCopy(const CallNode* call) {
    if (!call->op.same_as(DCUAsyncCopyOp()) || call->args.size() < 4 || !IsADcuRes(call->args[2])) {
      return GetRef<PrimExpr>(call);
    }

    auto [new_src_off, idxen] = SplitSrcOffset(call->args[3]);
    Array<PrimExpr> new_args;
    new_args.push_back(call->args[0]);
    new_args.push_back(call->args[1]);
    new_args.push_back(call->args[2]);
    new_args.push_back(new_src_off);
    new_args.push_back(idxen);
    // Preserve smem_offset (args[4]) and element dtype (args[5]) on tl.dcu_async_copy.
    for (size_t i = 4; i < call->args.size(); ++i) {
      new_args.push_back(call->args[i]);
    }
    return Call(call->dtype, DCUAsyncCopyIdxenOp(), new_args);
  }

  Stmt VisitStmt_(const LetStmtNode* op) final {
    PrimExpr value = VisitExpr(op->value);
    Stmt body = VisitStmt(op->body);
    return LetStmt(op->var, value, body);
  }

  Stmt VisitStmt_(const EvaluateNode* op) final {
    if (const auto* call = op->value.as<CallNode>()) {
      PrimExpr rewritten = RewriteDCUAsyncCopy(call);
      if (!rewritten.same_as(op->value)) {
        return Evaluate(rewritten);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    PrimExpr rewritten = RewriteDCUAsyncCopy(op);
    if (!rewritten.same_as(GetRef<PrimExpr>(op))) {
      return rewritten;
    }
    if (op->op.same_as(MakeDCUResourceOp())) {
      rewritten = RewriteMakeDCUResource(op);
      if (!rewritten.same_as(GetRef<PrimExpr>(op))) {
        return rewritten;
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  int64_t k_dim_;
  int64_t tile_width_;
  int struct_bit_;
  std::unordered_set<const VarNode*> a_dcu_res_vars_;
  arith::Analyzer analyzer_;
};

}  // namespace

PrimFunc InjectDCUAsyncCopyIdxenToFunc(PrimFunc func, const IRModule& mod) {
  std::optional<ADimInfo> a_info = ExtractADimInfoFromModule(mod);
  if (!a_info.has_value()) {
    return func;
  }

  const int64_t k_dim = *a_info->k_dim;
  const int elem_bytes = a_info->elem_bytes;
  ICHECK_GT(elem_bytes, 0);
  // StructBit template param: largest power-of-two less than K * elem_bytes
  // (byte stride along K). E.g. K=5120, int8 -> 4096 (bit=12); half -> 8192 (bit=13).
  const int64_t k_stride_bytes = k_dim * elem_bytes;
  const int struct_bit = StructBitFromTileWidth(TileWidthFromK(k_stride_bytes));
  // Idxen offset splitting still divides by K-scaled terms in element units;
  // codegen already accounts for sizeof(data) on the offset path.
  const int64_t tile_width = TileWidthFromK(k_dim);

  InfoCollector collector;
  collector(func->body);
  if (collector.a_dcu_res_vars.empty()) {
    return func;
  }

  class CopyChecker : public StmtExprVisitor {
   public:
    bool has_a_copy{false};

   private:
    void VisitExpr_(const CallNode* op) final {
      if (op->op.same_as(DCUAsyncCopyOp()) && op->args.size() >= 3) {
        if (const auto* var = op->args[2].as<VarNode>()) {
          if (a_vars_.count(var) > 0) {
            has_a_copy = true;
          }
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    }

   public:
    std::unordered_set<const VarNode*> a_vars_;
  } checker;
  checker.a_vars_ = collector.a_dcu_res_vars;
  checker(func->body);
  if (!checker.has_a_copy) {
    return func;
  }

  return IdxenInjector::Substitute(std::move(func), k_dim, tile_width, struct_bit,
                                   collector.a_dcu_res_vars);
}

namespace transform {

using namespace tir::transform;

tvm::transform::Pass InjectDCUAsyncCopyIdxen() {
  auto pass_func = [=](PrimFunc f, const IRModule& m, const PassContext& ctx) {
    if (!IsDCUTarget(m)) {
      return f;
    }
    return InjectDCUAsyncCopyIdxenToFunc(std::move(f), m);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectDCUAsyncCopyIdxen", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectDCUAsyncCopyIdxen", InjectDCUAsyncCopyIdxen);
}

}  // namespace transform
}  // namespace tl
}  // namespace tvm
