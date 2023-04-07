from typing import List, Mapping, Tuple
import os


def get_file_paths_helper(search_dir: str) -> List[Tuple[str, str]]:
    try:
        bins = filter(lambda x: x.is_dir(), os.scandir(search_dir))
        return list(map(lambda x: (x.name, os.path.join(search_dir, x.name)), bins))
    except:
        return []

def get_file_name_to_file_path_mapping(search_dir: str) -> Mapping[str, str]:
    paths = get_file_paths_helper(search_dir)
    return { name : p for [name, p] in paths }
