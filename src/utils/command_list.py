from typing import List
from command import *

SUPPORTED_COMMANDS: List[Command] = [
    Command('set', [
        CommandArg('max_gen_token_length', CommandArgType.INT, 100),
        CommandArg('top_k', CommandArgType.FLOAT),
        CommandArg('top_p', CommandArgType.FLOAT),
        CommandArg('temp', CommandArgType.FLOAT),
        CommandArg('repeat_penalty', CommandArgType.FLOAT),
        CommandArg('p_prefix', CommandArgType.STRING),
        CommandArg('p_suffix', CommandArgType.STRING),
    ]),
    Command('reset', []),
    Command('attach', [
        CommandArg('lora', CommandArgType.BOOLEAN, False),
    ]),
    Command('detach', [
        CommandArg('lora', CommandArgType.BOOLEAN, False),
    ]),
]