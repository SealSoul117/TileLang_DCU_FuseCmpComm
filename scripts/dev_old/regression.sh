#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
mkdir -p build/logs
python scripts/dev_old/check_paths.py | tee build/logs/path-isolation.json
python -m pytest -q \
    testing/python/distributed/test_m0_contracts.py \
    testing/python/distributed/test_m1_ir.py \
    testing/python/distributed/ir \
    testing/python/language/test_tilelang_language_frontend_v2.py::test_argument \
    testing/python/language/test_tilelang_language_frontend_v2.py::test_expr \
    testing/python/language/test_tilelang_language_frontend_v2.py::test_dtype_str_repr \
    testing/python/language/test_tilelang_language_frontend_v2.py::test_marco_return \
    testing/python/language/test_tilelang_language_frontend_v2.py::test_prim_func_generator \
    testing/python/transform/test_tilelang_transform_let_inline.py \
    testing/python/intrinsics/test_dcu_rewind_layout.py \
    --junitxml=build/old-upgrade-regression.xml \
    2>&1 | tee build/logs/regression.log
python scripts/dev_old/gemm_smoke.py 2>&1 | tee build/logs/gemm-smoke.log
