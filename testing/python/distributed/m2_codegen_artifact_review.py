"""Export real M2 device code for independent compiler/LLVM inspection.

This generates review artifacts; it does not replace the four-PE execution
tests or modify generated HIP to make it compile.
"""
from pathlib import Path
import os
import re
import subprocess

import tilelang
from tilelang.distributed.ipc_binding import bind_ipc
from test_m2_independent_review import primitive_compile_function


root = Path(__file__).resolve().parents[3]
output = root / "build-old" / "m2-codegen-review"
output.mkdir(parents=True, exist_ok=True)
artifact = tilelang.lower(
    bind_ipc(primitive_compile_function(), ["src", "dst", "flag"]),
    target="hip -mcpu=gfx936", enable_device_compile=False,
    enable_host_codegen=False,
)
source = output / "primitives.hip"
source.write_text(artifact.kernel_source)
llvm = output / "primitives.ll"
command = [
    str(Path(os.environ["ROCM_PATH"]) / "llvm/bin/clang++"),
    "-x", "hip", "--offload-arch=gfx936", "--cuda-device-only",
    "-std=c++17", "-O3", "-S", "-emit-llvm",
    "--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11",
    "-I" + str(root / "src"),
    "-I" + str(root / "src/runtime/dcu_ipc/include"),
    "-I" + str(root / "3rdparty/composable_kernel/include"),
    str(source), "-o", str(llvm),
]
subprocess.run(command, check=True)
text = llvm.read_text()
assert "__syncthreads();" in artifact.kernel_source
assert "fence seq_cst" in text
assert re.search(r"call void @llvm\.amdgcn\.s\.barrier\(\)", text)
assert re.search(r"store atomic[^\n]+release", text)
assert re.search(r"load atomic[^\n]+acquire", text)
print(f"PASS generated HIP -> LLVM: {llvm}")
