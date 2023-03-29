from enum import Enum
from typing import Callable, List, Mapping, Optional
from cpuinfo import get_cpu_info

cmake_variable_type = str

CMAKE_FEATURE_FILE_PATH = "./cmake/CompilerFlagVariables.cmake"

def save_cmake_vars(var_map: Mapping[str, List[str]]) -> None:
    with open(CMAKE_FEATURE_FILE_PATH, "w") as f:
        for k, v in var_map.items():
            s = ' '.join(set(v))
            f.write(f'set({k} "{s}")\n')

def get_gcc_flag(feature: str) -> Optional[str]:
    if 'fma' in feature:
        return '-mfma'
    elif 'f16c' == feature:
        return '-mf16c'
    elif 'avx2' == feature:
        return '-mavx2'
    elif 'avx1.0' == feature or 'avx' == feature:
        return '-mavx'
    elif 'sse3' == feature:
        return '-msse3'
    elif 'avx512f' == feature:
        return '-mavx512f'
    elif 'avx512bw' == feature:
        return '-mavx512bw'
    elif 'avx512dq' == feature:
        return '-mavx512dq'
    elif 'avx512vl' == feature:
        return '-mavx512vl'
    elif 'avx512cd' == feature:
        return '-mavx512cd'
    elif 'avx512er' == feature:
        return '-mavx512er'
    elif 'avx512ifma' == feature:
        return '-mavx512ifma'
    elif 'avx512pf' == feature:
        return '-mavx512pf'
    return None

def get_clang_flag(feature: str) -> Optional[str]:
    return "-march=native"

def get_msvc_flag(feature: str) -> Optional[str]:
    if 'fma' in feature:
        return None
    elif 'f16c' == feature:
        return None
    elif 'avx2' == feature:
        return '/arch:AVX2'
    elif 'avx1.0' == feature or 'avx' == feature:
        return '/arch:AVX'
    elif 'sse2' == feature:
        return '/arch:SSE2'
    elif 'avx512f' == feature:
        return '/arch:AVX512'
    elif 'avx512bw' == feature:
        return '/arch:AVX512'
    elif 'avx512dq' == feature:
        return '/arch:AVX512'
    elif 'avx512vl' == feature:
        return '/arch:AVX512'
    elif 'avx512cd' == feature:
        return '/arch:AVX512'
    elif 'avx512er' == feature:
        return '/arch:AVX512'
    elif 'avx512ifma' == feature:
        return '/arch:AVX512'
    elif 'avx512pf' == feature:
        return '/arch:AVX512'
    return None

COMPILER_LOOKUP_TABLE: Mapping[cmake_variable_type, Callable[[str], Optional[str]]] = {
    'GCC_CXXFLAG': get_gcc_flag,
    'CLANG_CXXFLAG': get_clang_flag,
    'MSVC_CXXFLAG': get_msvc_flag
}

def init_cmake_vars(cmake_var: str, arch: str) -> List[str]:
    if 'MSVC' in cmake_var:
        return ['/GL']
    return []

def get_compiler_flag(feature: str) -> Mapping[cmake_variable_type, Optional[str]]:
    return { v : c(feature)  for v, c in COMPILER_LOOKUP_TABLE.items()}

def main() -> None:
    info = get_cpu_info()
    arch: str = info['arch'] if 'arch' in info else ''
    cmake_vars: Mapping[cmake_variable_type, List[str]] = { v : init_cmake_vars(v, arch.upper()) for v in COMPILER_LOOKUP_TABLE.keys() }
    
    if 'flags' in info:
        flags: List[str] = info['flags']

        for f in flags:
            temp = get_compiler_flag(f.lower())
            for k, v in temp.items():
                if v is not None:
                    cmake_vars[k].append(v)
    save_cmake_vars(cmake_vars)

if __name__ == "__main__":
    main()