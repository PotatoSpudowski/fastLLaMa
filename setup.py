from dataclasses import dataclass
import os
import sys
import subprocess
from typing import Callable, List, Mapping, MutableMapping, Optional, Tuple, Union, cast
from cpuinfo import get_cpu_info
from scripts.utils.paths import get_file_name_to_file_path_mapping
from scripts.utils.shell import get_python_info, run_shell, select_language
import argparse

cmake_variable_type = str
CmakeVarType = MutableMapping[str, Union[List[str], bool, str]]

CMAKE_FEATURE_FILE_PATH = os.path.join('.', "cmake","CompilerFlagVariables.cmake")

ALL_LANGUAGES_IN_INTERFACES_PATH = os.path.join('.', 'interfaces')
ALL_LANGUAGES_IN_INTERFACES = get_file_name_to_file_path_mapping(ALL_LANGUAGES_IN_INTERFACES_PATH)

g_selected_language: Tuple[str, str] = ('c', os.path.join('.', ALL_LANGUAGES_IN_INTERFACES_PATH, 'c'))

def save_cmake_vars_helper(filepath: str, var_map: Mapping[str, Union[List[str], bool, str]]) -> None:
    with open(filepath, "w") as f:
        for k, v in var_map.items():
            if (type(v) == bool):
                f.write(f"set({k} {'TRUE' if v else 'FALSE'})\n")
            elif (type(v) == str):
                f.write(f'set({k} "{v}")\n')
            else:
                s = ';'.join(set(v))
                f.write(f'set({k} "{s}")\n')

def save_cmake_vars(var_map: Mapping[str, Union[List[str], bool]]) -> None:
    save_cmake_vars_helper(CMAKE_FEATURE_FILE_PATH, var_map)

def set_python_version(cmake_global_vars: CmakeVarType) -> None:
    python_info = get_python_info(os.path.join(',', 'scripts', 'utils', 'python_info.py'))

    if python_info is not None:
        cmake_global_vars['PYTHON_EXECUTABLE'] = [python_info.binary_path]
        cmake_global_vars['PYTHON_INCLUDE_DIR'] = [python_info.include_path]
        cmake_global_vars['PYTHON_LIBRARY'] = [python_info.library_path]
        print(f"Set python to '{python_info.binary_path}'")
    else:
        print("Auto detecting python version")

def set_global_cmake_variables(cmake_global_vars: CmakeVarType, args: argparse.Namespace) -> None:
    global g_selected_language

    def_lang_path = ALL_LANGUAGES_IN_INTERFACES['c']
    g_selected_language = ('c', def_lang_path)
    if args.gui:
        g_selected_language = select_language(ALL_LANGUAGES_IN_INTERFACES_PATH)
        # if g_selected_language[0] == 'python':
        #     set_python_version(cmake_global_vars)
    elif args.language:
        g_selected_language = (args.language, ALL_LANGUAGES_IN_INTERFACES[args.language])
    
    cmake_global_vars[f'EXAMPLES_{g_selected_language[0]}'] = True
    cmake_global_vars[f'INTERFACES_{g_selected_language[0]}'] = True
    cmake_global_vars['WORKSPACE'] = os.getcwd()

def run_make(build_dir: str = "build") -> None:
    # Change the current working directory to the build directory
    if not os.path.exists(build_dir):
       os.mkdir(build_dir)
    
    current_dir = os.getcwd()
    os.chdir(build_dir)

    try:
        # Run the 'make' command
        run_shell([
            'cmake ..',
            'make'
        ])
    except subprocess.CalledProcessError as e:
        print("An error occurred while running 'make':", e)
        print("Output:", e.output)
    finally:
        # Change back to the original working directory
        os.chdir(current_dir)

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
    return get_gcc_flag(feature)

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

def match_any(sub: List[str], string: str, match_sub = False) -> bool:
    for s in sub:
        if (not match_sub and s == string) or (match_sub and s in string):
            return True 
    return False

def fix_gcc_flags(flags: List[str]) -> List[str]:
    return flags

def fix_clang_flags(flags: List[str]) -> List[str]:
    return flags

def fix_msvc_flag(flags: List[str]) -> List[str]:
    return flags

COMPILER_LOOKUP_TABLE: Mapping[cmake_variable_type, Callable[[str], Optional[str]]] = {
    'GCC_CXXFLAG': get_gcc_flag,
    'CLANG_CXXFLAG': get_clang_flag,
    'MSVC_CXXFLAG': get_msvc_flag
}

COMPILER_FLAG_FIX_LOOKUP_TABLE: Mapping[cmake_variable_type, Callable[[List[str]], List[str]]] = {
    'GCC_CXXFLAG': fix_gcc_flags,
    'CLANG_CXXFLAG': fix_clang_flags,
    'MSVC_CXXFLAG': fix_msvc_flag
}

def init_cmake_vars(cmake_var: str, arch: str) -> List[str]:
    if 'MSVC' in cmake_var:
        return ['/GL']
    return []

def get_compiler_flag(feature: str) -> Mapping[cmake_variable_type, Optional[str]]:
    return { v : c(feature)  for v, c in COMPILER_LOOKUP_TABLE.items()}

def fix_flags(vars: MutableMapping[cmake_variable_type, List[str]]) -> None:
    for v, flags in vars.items():
        if v in COMPILER_FLAG_FIX_LOOKUP_TABLE:
            vars[v] = COMPILER_FLAG_FIX_LOOKUP_TABLE[v](flags)

def generate_compiler_flags() -> None:
    info = get_cpu_info()
    arch: str = info['arch'] if 'arch' in info else ''
    cmake_vars: MutableMapping[cmake_variable_type, List[str]] = { v : init_cmake_vars(v, arch.upper()) for v in COMPILER_LOOKUP_TABLE.keys() }
    
    if 'flags' in info:
        flags: List[str] = info['flags']

        for f in flags:
            temp = get_compiler_flag(f.lower())
            for k, v in temp.items():
                if v is not None:
                    cmake_vars[k].append(v)
    fix_flags(cmake_vars)
    save_cmake_vars(cmake_vars)

def run_cmd_on_build_dirs(cmd: List[List[str] | str]) -> None:
    example_paths = [os.path.join('.', 'examples', l) for l in ALL_LANGUAGES_IN_INTERFACES.keys()]
    current_path = os.getcwd()
    for path in example_paths:
        build_path = os.path.join(path, 'build')
        if os.path.exists(build_path):
            os.chdir(build_path)
            try:
                run_shell(cmd)
            finally:
                os.chdir(current_path)
    if os.path.exists(os.path.join('.', 'build')):
        os.chdir(os.path.join('.', 'build'))
        try:
            run_shell(cmd)
        finally:
            os.chdir(current_path)

def set_cc_android_flags(cmake_vars: CmakeVarType, args: argparse.Namespace) -> None:
    ndk = args.android_ndk
    abi = args.android_abi
    mode = args.android_mode
    version = args.android_platform
    neon = True if (args.android_neon is None and version > 23 and abi == 'v7') or args.android_neon else False
    use_lld = args.android_ld
    stl = args.android_stl

    cmake_vars['CMAKE_TOOLCHAIN_FILE'] = ndk

    if abi == 'v7':
        cmake_vars['ANDROID_ABI'] = 'armeabi-v7a'
    elif abi == 'v8':
        cmake_vars['ANDROID_ABI'] = 'arm64-v8a'
    else:
        cmake_vars['ANDROID_ABI'] = abi

    cmake_vars['ANDROID_ARM_MODE'] = mode
    cmake_vars['ANDROID_ARM_NEON'] = neon
    cmake_vars['ANDROID_PLATFORM'] = f'android-{version}'
    
    if stl == 'shared':
        cmake_vars['ANDROID_STL'] = 'c++_shared'
    elif stl == 'static':
        cmake_vars['ANDROID_STL'] = 'c++_static'
    else: 
        cmake_vars['ANDROID_STL'] = stl

    if use_lld:
        cmake_vars['ANDROID_LD'] = True


def set_cross_compile_target_flags(cmake_vars: CmakeVarType, args: argparse.Namespace) -> None:
    cc_target = args.cc_target

    if cc_target == 'android':
        set_cc_android_flags(cmake_vars, args)
    # if cc_target == 'android':

def set_android_arg_parser(parser: argparse._SubParsersAction) -> None:
    android_parser: argparse.ArgumentParser = parser.add_parser('android', help="Android")
    android_parser.add_argument("-ndk", help="Path to the android NDK", required=True, dest="android_ndk")
    
    android_parser.add_argument("-abi",
        choices=['v7', 'v8', 'x86', 'x86_64'],
        help="Select the android ABI. v7='armeabi-v7a', v8='arm64-v8a'. Default is 'v7'",
        default="v7",
        dest="android_abi"
    )

    android_parser.add_argument(
        "-mode",
        help="Specifies whether to generate arm or thumb instructions for armeabi-v7a. Default is 'thumb'",
        choices=['arm', 'thumb'],
        default='thumb',
        dest="android_mode"
    )
    
    android_parser.add_argument(
        "-neon",
        help="Enables or disables NEON for armeabi-v7a",
        default=None,
        action='store_true',
        dest="android_neon"
    )

    android_parser.add_argument(
        "-lld",
        help="Use lld to link",
        action='store_true',
        dest="android_ld"
    )

    android_parser.add_argument("-stl",
        choices=['shared', 'static', 'none', 'system'],
        help="Specifies which STL to use for this application. shared='c++_shared', static='c++_static'. Default is 'static'",
        default="static",
        dest="android_stl"
    )

    android_parser.add_argument("-platform",
        help="Specifies the minimum API level supported by the application or library. Default is '23'",
        default=23,
        dest="android_platform",
        type=int
    )


def parse_args() -> bool:
    parser = argparse.ArgumentParser(
        prog="Fastllama",
        description="Fastllama tries to provide llama wrapper interfaces for all popular languages."
    )
    parser.add_argument('-l', '--language', choices=ALL_LANGUAGES_IN_INTERFACES.keys(), default='c', help="Select a project language. Default is 'c'")
    parser.add_argument('-g', '--gui', action='store_true', help="Select a project language using GUI.")
    parser.add_argument('-c', '--clean', action='store_true', help="This command is equivalent to 'make clean'")
    parser.add_argument('-m', '--make', action='store_true' , help="This command is equivalent to 'make'. This avoids complete rebuild process.")
    cc_subparser = parser.add_subparsers(help="Cross compilation help", dest="cc_target")

    set_android_arg_parser(cc_subparser)

    # parser.add_argument('-cc', '--cross-compile', choices=['android'], help="Cross compile for specific operating system", default=None)

    args = parser.parse_args(sys.argv[1:])
    if args.clean:
        run_cmd_on_build_dirs([['make', 'clean']])
        return False
    if args.make:
        run_cmd_on_build_dirs(['make'])
        return False
    
    cmake_global_vars: CmakeVarType = {}
    set_global_cmake_variables(cmake_global_vars, args)
    set_cross_compile_target_flags(cmake_global_vars, args)
    save_cmake_vars_helper(os.path.join('.', 'cmake', 'GlobalVars.cmake'), cmake_global_vars)
    return True

def build_example() -> None:
    global g_selected_language
    example_path = os.path.join('.', 'examples', g_selected_language[0])
    if not os.path.exists(example_path):
        return
    
    if os.path.exists(os.path.join(example_path, 'CMakeLists.txt')):
        print('\n\nBuilding Examples....\n')
        current_dir = os.getcwd()
        os.chdir(example_path)
        try:
            run_make()
        finally:
            os.chdir(current_dir)
            print('\nBuilding examples completed\n')
    if g_selected_language[0] == 'python':
        module_path = os.path.join('.', 'interfaces', 'python', 'fastllama.py')
        lib_path = os.path.join(example_path, 'build', 'fastllama.py')
        if not os.path.exists(os.path.join(example_path, 'build')):
            os.mkdir(os.path.join(example_path, 'build'))
        if os.path.exists(lib_path):
            return
        # with open(os.path.join('.', 'interfaces', 'python', 'lib_path.py'), 'wt') as f:
        #     f.write(f'WORKSPACE={os.getcwd()}')
        os.chmod(module_path, 0o700)
        os.symlink(os.path.abspath(module_path), os.path.abspath(lib_path))

def main() -> None:
    if not parse_args():
        return
    generate_compiler_flags()
    run_make()
    build_example()

if __name__ == "__main__":
    main()