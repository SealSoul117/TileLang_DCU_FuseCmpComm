/*!
 * \file inject_dcu_async_copy_idxen.cc
 * \brief Convert tl.dcu_async_copy to tl.dcu_async_copy_idxen with structured
 *        buffer resource for DCU global-to-shared loads.
 *
 * A-matrix idxen applies when A is not transposed; B-matrix idxen when B is transposed.
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

bool IsBBufferName(const std::string& name) {
  return name == "B" || name == "B_1";
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
  while ((p << 1) <= k) {
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

struct GemmDimInfo {
  std::optional<int64_t> a_k_dim;
  std::optional<int64_t> b_k_dim;
  int a_elem_bytes{1};
  int b_elem_bytes{1};
};

std::optional<int64_t> ExtractKDimFromBuffer(const Buffer& buffer, bool transposed, bool is_a = true) {
  if (buffer->shape.size() != 2) {
    return std::nullopt;
  }
  const int k_axis = (transposed == is_a) ? 0 : 1;
  if (const auto* imm = buffer->shape[k_axis].as<IntImmNode>()) {
    return imm->value;
  }
  return std::nullopt;
}

class GemmDimCollector : public StmtExprVisitor {
 public:
  GemmDimInfo info;

 private:
  void VisitStmt_(const DeclBufferNode* op) final {
    if (IsABufferName(op->buffer->name)) {
      info.a_elem_bytes = op->buffer->dtype.bytes();
      if (auto k_dim = ExtractKDimFromBuffer(op->buffer, /*transposed=*/false)) {
        info.a_k_dim = k_dim;
      }
    } else if (IsBBufferName(op->buffer->name)) {
      info.b_elem_bytes = op->buffer->dtype.bytes();
      if (auto k_dim = ExtractKDimFromBuffer(op->buffer, /*transposed=*/true, false)) {
        info.b_k_dim = k_dim;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

std::optional<GemmDimInfo> ExtractGemmDimInfoFromModule(const IRModule& mod) {
  for (const auto& kv : mod->functions) {
    if (const auto* func = kv.second.as<PrimFuncNode>()) {
      GemmDimCollector collector;
      collector(func->body);
      if (collector.info.a_k_dim.has_value() || collector.info.b_k_dim.has_value()) {
        return collector.info;
      }
    }
  }
  return std::nullopt;
}

void FillGemmDimInfoFromBufferMap(const PrimFunc& func, bool trans_a, bool trans_b,
                                  GemmDimInfo* info) {
  for (const auto& kv : func->buffer_map) {
    const Buffer& buffer = kv.second;
    if (IsABufferName(buffer->name)) {
      info->a_elem_bytes = buffer->dtype.bytes();
      if (auto k_dim = ExtractKDimFromBuffer(buffer, trans_a)) {
        info->a_k_dim = k_dim;
      }
    } else if (IsBBufferName(buffer->name)) {
      info->b_elem_bytes = buffer->dtype.bytes();
      if (auto k_dim = ExtractKDimFromBuffer(buffer, trans_b, false)) {
        info->b_k_dim = k_dim;
      }
    }
  }
}

struct IdxenMatrixParams {
  bool enabled{false};
  int64_t k_dim{0};
  int64_t tile_width{0};
  int struct_bit{0};
  int elem_bytes{0};
  std::unordered_set<const VarNode*> dcu_res_vars;
};

class InfoCollector : public StmtExprVisitor {
 public:
  std::unordered_set<const VarNode*> a_dcu_res_vars;
  std::unordered_set<const VarNode*> b_dcu_res_vars;

 private:
  void VisitStmt_(const LetStmtNode* op) final {
    if (const auto* call = op->value.as<CallNode>()) {
      if (call->op.same_as(MakeDCUResourceOp()) && call->args.size() >= 1) {
        const std::string global_name = GetGlobalBufferName(call->args[0]);
        if (IsABufferName(global_name)) {
          a_dcu_res_vars.insert(op->var.get());
        } else if (IsBBufferName(global_name)) {
          b_dcu_res_vars.insert(op->var.get());
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

// Match FloorMod(x, m) or (x & (m-1)) with m > 1. Returns (x, m).
std::optional<std::pair<PrimExpr, int64_t>> MatchBoundedIndex(const PrimExpr& expr) {
  PrimExpr e = StripCast(expr);
  if (const auto* fm = e.as<FloorModNode>()) {
    if (const auto* imm = fm->b.as<IntImmNode>()) {
      if (imm->value > 1) {
        return std::make_pair(fm->a, imm->value);
      }
    }
  }
  if (const auto* call = e.as<CallNode>()) {
    if (call->op.same_as(builtin::bitwise_and()) && call->args.size() == 2) {
      const IntImmNode* mask = call->args[1].as<IntImmNode>();
      PrimExpr x = call->args[0];
      if (mask == nullptr) {
        mask = call->args[0].as<IntImmNode>();
        x = call->args[1];
      }
      if (mask != nullptr && mask->value > 0) {
        const int64_t m = mask->value + 1;
        if ((m & (m - 1)) == 0) {
          return std::make_pair(x, m);
        }
      }
    }
  }
  return std::nullopt;
}

// (x % m) * coeff can exceed one K-stride even when coeff itself is not a
// multiple of K, e.g. (tx % 8) * 8 with K=32:
//   (tx % 8) * 8 = ((tx % 8) >> 2) * 32 + (tx % 4) * 8
// Put the high part into idxen so offen stays strictly inside one record.
std::optional<std::pair<PrimExpr, PrimExpr>> TrySplitOverflowModTerm(const PrimExpr& term,
                                                                     int64_t k_dim) {
  PrimExpr e = StripCast(term);
  const auto* mul = e.as<MulNode>();
  if (mul == nullptr) {
    return std::nullopt;
  }

  const IntImmNode* imm = mul->a.as<IntImmNode>();
  PrimExpr other = mul->b;
  if (imm == nullptr) {
    imm = mul->b.as<IntImmNode>();
    other = mul->a;
  }
  if (imm == nullptr || k_dim <= 0) {
    return std::nullopt;
  }
  const int64_t coeff = imm->value;
  if (coeff == 0) {
    return std::nullopt;
  }
  const int64_t acoeff = coeff < 0 ? -coeff : coeff;
  if (k_dim % acoeff != 0) {
    return std::nullopt;
  }
  const int64_t units = k_dim / acoeff;
  auto bounded = MatchBoundedIndex(other);
  if (!bounded.has_value() || bounded->second <= units) {
    return std::nullopt;
  }

  const PrimExpr& x = bounded->first;
  const int64_t m = bounded->second;
  const DataType dt = e.dtype();
  PrimExpr idxen =
      floordiv(floormod(x, make_const(x.dtype(), m)), make_const(x.dtype(), units));
  if (coeff < 0) {
    idxen = -idxen;
  }
  if (idxen.dtype() != dt) {
    idxen = Cast(dt, idxen);
  }
  PrimExpr rest = floormod(x, make_const(x.dtype(), units)) * make_const(dt, coeff);
  return std::make_pair(idxen, rest);
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

IdxenMatrixParams BuildMatrixParams(bool enabled, int64_t k_dim, int elem_bytes,
                                    const std::unordered_set<const VarNode*>& dcu_res_vars) {
  IdxenMatrixParams params;
  params.enabled = enabled;
  params.dcu_res_vars = dcu_res_vars;
  if (!enabled || dcu_res_vars.empty() || k_dim <= 0) {
    params.enabled = false;
    return params;
  }
  ICHECK_GT(elem_bytes, 0);
  const int64_t k_stride_bytes = k_dim * elem_bytes;
  params.k_dim = k_dim;
  params.struct_bit = StructBitFromTileWidth(TileWidthFromK(k_stride_bytes));
  params.tile_width = TileWidthFromK(k_stride_bytes) / elem_bytes;
  params.elem_bytes = elem_bytes;
  return params;
}

class IdxenInjector : public StmtExprMutator {
 public:
  IdxenInjector(IdxenMatrixParams a_params, IdxenMatrixParams b_params)
      : a_params_(std::move(a_params)), b_params_(std::move(b_params)) {}

  static PrimFunc Substitute(PrimFunc func, const IdxenMatrixParams& a_params,
                             const IdxenMatrixParams& b_params) {
    IdxenInjector mutator(a_params, b_params);
    auto* n = func.CopyOnWrite();
    n->body = mutator(std::move(n->body));
    return func;
  }

 private:
  const IdxenMatrixParams* ParamsForDcuRes(const PrimExpr& expr) const {
    if (const auto* var = expr.as<VarNode>()) {
      if (a_params_.enabled && a_params_.dcu_res_vars.count(var) > 0) {
        return &a_params_;
      }
      if (b_params_.enabled && b_params_.dcu_res_vars.count(var) > 0) {
        return &b_params_;
      }
    }
    return nullptr;
  }

  // Split offset = sum(K-scaled terms) + sum(rest) into:
  //   idxen = sum(term / K) for terms whose coefficient is a multiple of K
  //           plus high bits of (x % m) * coeff when that product can exceed K
  //   new_offset = idxen * (K - tile_width) + sum(rest)  with rest < K
  std::pair<PrimExpr, PrimExpr> SplitSrcOffset(const PrimExpr& offset, const IdxenMatrixParams& params) {
    auto split_scalar = [&](const PrimExpr& base) -> std::pair<PrimExpr, PrimExpr> {
      std::vector<PrimExpr> terms;
      CollectAddTerms(base, &terms);

      PrimExpr idxen = make_zero(base.dtype());
      PrimExpr offset_rest = make_zero(base.dtype());
      for (const PrimExpr& term : terms) {
        if (auto idxen_term = TryExtractIdxenTerm(term, params.k_dim)) {
          idxen = idxen + idxen_term.value();
        } else if (auto overflow = TrySplitOverflowModTerm(term, params.k_dim)) {
          idxen = idxen + overflow->first;
          offset_rest = offset_rest + overflow->second;
        } else {
          offset_rest = offset_rest + term;
        }
      }

      idxen = analyzer_.Simplify(idxen);
      offset_rest = analyzer_.Simplify(offset_rest);
      const int64_t k_residual = params.k_dim - params.tile_width;
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
    const std::string global_name = GetGlobalBufferName(call->args[0]);
    const IdxenMatrixParams* params = nullptr;
    if (IsABufferName(global_name)) {
      params = a_params_.enabled ? &a_params_ : nullptr;
    } else if (IsBBufferName(global_name)) {
      params = b_params_.enabled ? &b_params_ : nullptr;
    }
    if (params == nullptr) {
      return GetRef<PrimExpr>(call);
    }
    Array<PrimExpr> new_args = call->args;
    new_args.push_back(IntImm(DataType::Int(32), params->struct_bit));
    return Call(call->dtype, call->op, new_args);
  }

  PrimExpr RewriteDCUAsyncCopy(const CallNode* call) {
    if (!call->op.same_as(DCUAsyncCopyOp()) || call->args.size() < 4) {
      return GetRef<PrimExpr>(call);
    }
    const IdxenMatrixParams* params = ParamsForDcuRes(call->args[2]);
    if (params == nullptr) {
      return GetRef<PrimExpr>(call);
    }

    auto [new_src_off, idxen] = SplitSrcOffset(call->args[3], *params);
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

  IdxenMatrixParams a_params_;
  IdxenMatrixParams b_params_;
  arith::Analyzer analyzer_;
};

bool HasAsyncCopyForVars(const Stmt& body, const std::unordered_set<const VarNode*>& vars) {
  if (vars.empty()) {
    return false;
  }
  class CopyChecker : public StmtExprVisitor {
   public:
    bool has_copy{false};

   private:
    void VisitExpr_(const CallNode* op) final {
      if (op->op.same_as(DCUAsyncCopyOp()) && op->args.size() >= 3) {
        if (const auto* var = op->args[2].as<VarNode>()) {
          if (vars_.count(var) > 0) {
            has_copy = true;
          }
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    }

   public:
    std::unordered_set<const VarNode*> vars_;
  } checker;
  checker.vars_ = vars;
  checker(body);
  return checker.has_copy;
}

}  // namespace

PrimFunc InjectDCUAsyncCopyIdxenToFunc(PrimFunc func, const IRModule& mod) {
  DCUGemmTransposeFlags flags = GetDCUGemmTransposeFlags(func);
  DCUCopyBufferMaps maps = GetDCUCopyBufferMaps(func);
  const bool enable_a = NeedsDCUAsyncCopyIdxenForOperand(
      flags, 0, LookupDCUNeedLayoutForOperand(maps, 0, true));
  const bool enable_b = NeedsDCUAsyncCopyIdxenForOperand(
      flags, 1, LookupDCUNeedLayoutForOperand(maps, 1, true));
  if (!enable_a && !enable_b) {
    return func;
  }

  GemmDimInfo dim_info;
  if (auto module_info = ExtractGemmDimInfoFromModule(mod)) {
    dim_info = *module_info;
  }
  FillGemmDimInfoFromBufferMap(func, flags.trans_a, flags.trans_b, &dim_info);

  InfoCollector collector;
  collector(func->body);

  IdxenMatrixParams a_params =
      BuildMatrixParams(enable_a, dim_info.a_k_dim.value_or(0), dim_info.a_elem_bytes,
                        collector.a_dcu_res_vars);
  IdxenMatrixParams b_params =
      BuildMatrixParams(enable_b, dim_info.b_k_dim.value_or(0), dim_info.b_elem_bytes,
                        collector.b_dcu_res_vars);

  if (!a_params.enabled && !b_params.enabled) {
    return func;
  }

  const bool has_a_copy = a_params.enabled && HasAsyncCopyForVars(func->body, a_params.dcu_res_vars);
  const bool has_b_copy = b_params.enabled && HasAsyncCopyForVars(func->body, b_params.dcu_res_vars);
  if (!has_a_copy && !has_b_copy) {
    return func;
  }

  if (has_a_copy) {
    ICHECK(dim_info.a_k_dim.has_value()) << "InjectDCUAsyncCopyIdxen: missing A K dimension";
  }
  if (has_b_copy) {
    ICHECK(dim_info.b_k_dim.has_value()) << "InjectDCUAsyncCopyIdxen: missing B K dimension";
  }

  if (!has_a_copy) {
    a_params.enabled = false;
  }
  if (!has_b_copy) {
    b_params.enabled = false;
  }

  return IdxenInjector::Substitute(std::move(func), a_params, b_params);
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
