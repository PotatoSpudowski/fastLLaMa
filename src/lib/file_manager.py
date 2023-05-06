from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import os
from typing import List, Union

def workspace_path():
    return Path(os.path.dirname(os.path.realpath(__file__))) / '..' / '..'

class FileKind(Enum):
    FILE = 1
    DIR = 2

@dataclass
class FileStructure:
    path: Path
    file_name: str
    kind: FileKind

    def get_path(self) -> Path:
        return self.path
    
    def get_file_name(self) -> str:
        return self.file_name
    
    def get_file(self) -> Path:
        return self.path / self.file_name
    
    def is_dir(self) -> bool:
        return self.kind == FileKind.DIR
    
    def is_file(self) -> bool:
        return self.kind == FileKind.FILE
    
    def to_json(self) -> dict:
        return {
            'path': str(self.get_file()),
            'file_name': self.get_file_name(),
            'kind': 'file' if self.is_file() else 'directory'
        }

class FileManager:
    def __init__(self, path: Union[str, os.PathLike[str]] = workspace_path()):
        self.path = Path(path)

    def get_path(self) -> Path:
        return self.path
    
    def get_file(self, file_name) -> Path:
        return self.path / file_name
    
    def get_file_list(self) -> List[FileStructure]:
        files: List[FileStructure] = []
        for file in os.listdir(self.path):
            if os.path.isfile(self.path / file):
                files.append(FileStructure(self.path, file, FileKind.FILE))
            if os.path.isdir(self.path / file):
                files.append(FileStructure(self.path, file, FileKind.DIR))
        return files
    def go_back(self):
        self.path = self.path.parent
    
    def open_dir(self, dir_name):
        if not os.path.isdir(self.path / dir_name):
            raise Exception("Not a directory")
        self.path = self.path / dir_name

    def to_json(self) -> dict:
        return {
            'path': str(self.get_path()),
            'files': [file.to_json() for file in self.get_file_list()]
        }
