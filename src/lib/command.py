from enum import Enum
from typing import Optional, Union


class CommandArgType(Enum):
    BOOLEAN = 1
    STRING = 2
    INT = 3
    FLOAT = 4

class CommandArg:
    def __init__(self, name: str, type: CommandArgType, value: Union[str, int, float, bool, None]=None):
        self.name = name
        self.type = type
        self.value = value

    def __str__(self):
        return f"{self.name}[T={self.type}]={self.value}"

    def __repr__(self):
        return str(self)
    
    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, CommandArg):
            return False
        return self.name == __value.name
    
    def to_json(self):
        return {
            'name': self.name,
            'type': self.type,
            'value': self.value,
        }
    
    def get_name(self):
        return self.name
    
    def get_type(self):
        return self.type
    
    def get_value(self):
        return self.value
    
    def is_string(self):
        return self.type == CommandArgType.STRING
    
    def is_int(self):
        return self.type == CommandArgType.INT
    
    def is_float(self):
        return self.type == CommandArgType.FLOAT
    
    def is_boolean(self):
        return self.type == CommandArgType.BOOLEAN
    
    def as_string(self):
        return str(self.value)
    
    def as_int(self):
        return int(self.value)
    
    def as_float(self):
        return float(self.value)
    
    def as_boolean(self):
        return bool(self.value)
    
    def is_none(self):
        return self.value is None

class Command:
    def __init__(self, name: str, args: list[CommandArg]):
        self.name = name
        self.args = { item.get_name() : item for item in args }

    def __str__(self):
        return f"{self.name} ({len(self.args)} {self.args.join(' ')})"
    
    def __repr__(self):
        return str(self)
    
    def to_json(self):
        return {
            'name': self.name,
            'args': [arg.to_json() for arg in self.args]
        }
    
    def get_name(self):
        return self.name
    
    def get_args(self):
        return self.args
    
    def get_arg(self, name: str) -> Optional[CommandArg]:
        return self.args.get(name, None)
    
    def get_arg_value(self, name: str) -> Optional[Union[str, int, float, bool]]:
        arg = self.get_arg(name)
        if arg is None:
            return None
        return arg.get_value()
    
    def get_arg_type(self, name: str) -> Optional[CommandArgType]:
        arg = self.get_arg(name)
        if arg is None:
            return None
        return arg.get_type()
    
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Command):
            return False
        return self.name == __value.name
    
    def __hash__(self) -> int:
        return hash(self.name)
