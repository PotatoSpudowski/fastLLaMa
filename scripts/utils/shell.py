import subprocess
from typing import List, Union

def run_shell(commands: List[Union[List[str], str]]) -> None:
    for cmd in commands:
        normalized_command = [cmd] if type(cmd) == str else cmd
        print(f"Setup executing command: '{subprocess.list2cmdline(normalized_command)}'")
        result = subprocess.Popen(normalized_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        while True:
            if result.stdout is None:
                break
            output = result.stdout.readline()
            if output:
                print(output, end='', flush=True)
            else:
                break