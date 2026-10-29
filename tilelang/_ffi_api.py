"""FFI APIs for tilelang"""

from tilelang._ffi_compat import init_ffi_api

# TVM_REGISTER_GLOBAL("tl.name").set_body_typed(func);
init_ffi_api("tl", __name__)
