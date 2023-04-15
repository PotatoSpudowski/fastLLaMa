from typing import List, MutableMapping
import os
import re

POSSIBLE_POSIX_PYTHON_PATHS: List[str] = [
    '/bin',
    '/etc',
    '/lib',
    '/lib/x86_64-linux-gnu',
    '/lib64',
    '/sbin',
    '/snap/bin',
    '/usr/bin',
    '/usr/games',
    '/usr/include',
    '/usr/lib',
    '/usr/lib/x86_64-linux-gnu',
    '/usr/lib64',
    '/usr/libexec',
    '/usr/local',
    '/usr/local/bin',
    '/usr/local/etc',
    '/usr/local/games',
    '/usr/local/lib',
    '/usr/local/sbin',
    '/usr/sbin',
    '/usr/share',
    '~/.local/bin',
]


def bin_paths() -> List[str]:
    if os.name == 'posix':
        return list(set(os.get_exec_path() + POSSIBLE_POSIX_PYTHON_PATHS))
    return os.get_exec_path()

PYTHON_SEARCH_PATTERN = re.compile(r"^python(\d+(\.\d+)?)?$")

def match_python_bin_file(filename: str) -> bool:
    """
        This Reg-ex matches following file names:
        python
        python3
        python38
        python3.8
    """
    basename = os.path.basename(filename)
    return PYTHON_SEARCH_PATTERN.search(basename) is not None

def find_python_binaries_in_dir(search_dir: str) -> List[str]:
    try:
        bins = filter(lambda x: not x.is_dir(), os.scandir(search_dir))
        full_path = map(lambda x: os.path.join(search_dir, x.name), bins)
        return list(filter(match_python_bin_file, full_path))
    except:
        return []

def pick_shortest_path(python_paths: List[str]) -> str:
    min_len = len(python_paths[0])
    path = python_paths[0]
    for p in python_paths:
        if min_len > len(p):
            min_len = len(p)
            path = p

    return path

def get_python_bin_from_paths(search_dirs: List[str]) -> List[str]:
    link_map: MutableMapping[str, List[str]] = {}
    for search_dir in search_dirs:
        paths = find_python_binaries_in_dir(search_dir)
        for path in paths:
            try:
                real_path = os.path.realpath(path)
                if real_path in link_map:
                    link_map[real_path].append(path)
                else:
                    link_map[real_path] = [path]
            except:
                """
                    Nothing to dO
                """
    return list(set([pick_shortest_path(ps) for _, ps in link_map.items()]))

def get_python_exec_paths() -> List[str]:
    return get_python_bin_from_paths(bin_paths())