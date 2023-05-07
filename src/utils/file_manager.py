from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import os
from typing import List, Optional, Tuple, Union

def workspace_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

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
            'name': self.get_file_name(),
            'kind': 'file' if self.is_file() else 'directory'
        }

class FileManager:
    def __init__(self, path: Union[str, os.PathLike[str]] = workspace_path()):
        self.path = Path(path)

    def get_path(self) -> str:
        return str(self.path)
    
    def get_file(self, file_name) -> str:
        return str(self.path / file_name)
    
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
    
    def open_dir(self, dir_name: Optional[str] = None, path: Optional[Union[str, os.PathLike[str]]] = None) -> None:
        if path is not None:
            self.path = Path(path)
        if dir_name is not None:
            if not os.path.isdir(self.path / dir_name):
                raise Exception(f"'{self.path/dir_name}' not a directory")
            self.path = self.path / dir_name

    def to_json(self) -> dict:
        return {
            'path': str(self.get_path()),
            'files': [file.to_json() for file in self.get_file_list()]
        }
    
    @staticmethod
    def is_valid(ws_message: dict) -> Tuple[bool, str]:
        if "path" not in ws_message:
            return (False, "'path' is required")
        if "type" not in ws_message:
            return (False, "'type' is required")
        if "kind" not in ws_message:
            return (False, "'kind' is required")

        if type(ws_message["path"]) != str:
            return (False, "'path' must be a string")

        if ws_message["kind"] == "open-dir":
            if os.path.exists(ws_message["path"]):
                if not os.path.isdir(ws_message["path"]):
                    return (False, "'path' must be a directory")
            else:
                return (False, f"'{ws_message['path']}' does not exist")
        elif ws_message["kind"] == "go-back":
            if os.path.exists(ws_message["path"]):
                if not os.path.isdir(ws_message["path"]):
                    return (False, "'path' must be a directory")
            else:
                return (False, f"'{ws_message['path']}' does not exist")
            
        return (True, "valid")
