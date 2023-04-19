from dataclasses import dataclass
import subprocess
import sys
from typing import List, Optional, Sequence, Tuple, Union
import inquirer
from scripts.utils.paths import get_file_name_to_file_path_mapping
from scripts.utils.python_version import get_python_exec_paths

def run_shell(commands: Sequence[Union[List[str], str]]) -> None:
    for cmd in commands:
        normalized_command = subprocess.list2cmdline([cmd] if type(cmd) == str else cmd)
        print(f"Setup executing command: {normalized_command}")
        result = subprocess.Popen([normalized_command], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        while True:
            if result.stdout is None:
                break
            output = result.stdout.readline()
            if output:
                print(output, end='', flush=True)
            else:
                break

def choose_option(name: str, message: str, options: List[str]) -> Optional[str]:
    questions = [
        inquirer.List(name, message=message, choices=options)
    ]
    answers = inquirer.prompt(questions)
    return answers[name]

@dataclass
class PythonInfo:
    binary_path: str
    include_path: str
    library_path: str

def get_python_info(info_script_path: str) -> Optional[PythonInfo]:
    print('', flush=True, end='')
    python_paths = get_python_exec_paths()
    python_paths.sort()

    python_version = choose_option('python_version', 'Select a python version', python_paths)
    if python_version is None:
        return None

    print(f"Selected python version: '{python_version}'")

    print("Getting python information, please wait...")

    result = subprocess.run([f'{python_version} {info_script_path}'], shell=True, text=True, capture_output=True)
    if result.returncode == 0:
        output = list(filter(lambda x: len(x.strip()) != 0, result.stdout.split(';')))
        if len(output) != 3:
            print(f"Error occurred while getting python information. Needed both include and library path, but got one of it", file=sys.stderr)
            return None
        return PythonInfo(binary_path=python_version, library_path=output[0], include_path=output[1])
    else:
        print(f"Error occurred while getting python information: code(='{result.returncode}'), ", result.stderr, file=sys.stderr)
        return None

def select_language(search_dir: str) -> Tuple[str, str]:
    all_langs = get_file_name_to_file_path_mapping(search_dir)
    maybe_lang = choose_option("lang", "Select language to compile", [x.capitalize() for x in all_langs.keys()])
    lang = maybe_lang.lower() if maybe_lang is not None else 'c'
    return (lang, all_langs[lang])
