"""Check that the source checkout and its default build are actually loaded."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / 'build'
TVM = ROOT / '3rdparty/tvm'


def check_paths():
    import tilelang
    import tvm
    from tilelang import libinfo
    from tilelang.env import env
    from tilelang._ffi_compat import _ffi

    paths = {
        'python': sys.executable,
        'tilelang': tilelang.__file__,
        'tvm': tvm.__file__,
        'ffi': getattr(_ffi, '__file__', None),
        'tilelang_module': libinfo.find_lib_path('tilelang_module'),
        'templates': str(env.TILELANG_TEMPLATE_PATH),
    }
    assert Path(paths['tilelang']).resolve().is_relative_to(ROOT / 'tilelang'), paths
    assert Path(paths['tvm']).resolve().is_relative_to(TVM / 'python'), paths
    assert Path(paths['tilelang_module']).resolve().is_relative_to(BUILD), paths
    assert Path(paths['templates']).resolve() == ROOT / 'src', paths
    maps = Path('/proc/self/maps')
    if maps.exists():
        ffi_libraries = {line.split()[-1] for line in maps.read_text().splitlines()
                         if '/' in line and 'libtvm_ffi.so' in line}
        assert len(ffi_libraries) <= 1, ffi_libraries
    report = {'status': 'PASS', 'paths': paths}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    check_paths()
