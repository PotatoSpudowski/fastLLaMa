from datetime import datetime
import json
from threading import Thread
import threading
from typing import Coroutine
from fastllama.api import ProgressTag
from starlette.websockets import WebSocket, WebSocketState
from fastllama import Model, Logger
from .message_manager import *
from .file_manager import *
from .command_list import *
import asyncio
import os
from typing import Optional, cast, Dict, List
from pathlib import Path

MODEL_SAVE_PATH = Path(workspace_path()) / 'saves'

class FastllamaWebsocketLogger(Logger):
    def __init__(self, socket: 'FastllamaWebsocket'):
        self.socket = socket
        self.progress_task: Dict[ProgressTag, SystemMessage] = {}
        self.mutex = threading.Lock()
    
    def log_info(self, func_name: str, message: str) -> None:
        sys_message = self.socket.message_manager.make_system_message(
            kind=SystemMessageKind.INFO,
            message=message,
            function=func_name
        )
        # self.socket.message_queue.put_nowait(sys_message.to_json())
        # self.socket.message_event.set()
        asyncio.run_coroutine_threadsafe(self.socket.send_system_message(sys_message), cast(asyncio.AbstractEventLoop, self.socket.high_priority_loop))
        super().log_info(func_name, message)
    
    def log_err(self, func_name: str, message: str) -> None:
        sys_message = self.socket.message_manager.make_system_message(
            kind=SystemMessageKind.ERROR,
            message=message,
            function=func_name
        )
        asyncio.run_coroutine_threadsafe(self.socket.send_system_message(sys_message), cast(asyncio.AbstractEventLoop, self.socket.high_priority_loop))
        super().log_err(func_name, message)
    
    def log_warn(self, func_name: str, message: str) -> None:
        sys_message = self.socket.message_manager.make_system_message(
            kind=SystemMessageKind.WARNING,
            message=message,
            function=func_name
        )
        asyncio.run_coroutine_threadsafe(self.socket.send_system_message(sys_message), cast(asyncio.AbstractEventLoop, self.socket.high_priority_loop))
        super().log_warn(func_name, message)
    
    def progress(self, tag: ProgressTag, done_size: int, total_size: int) -> None:
        if (tag != ProgressTag.Init):
            return
        if tag not in self.progress_task:
            with self.mutex:
                if tag not in self.progress_task:
                    self.progress_task[tag] = self.socket.message_manager.make_system_message(
                        kind=SystemMessageKind.PROGRESS,
                        message=f"loading {tag}",
                        function=f"{tag}"
                    )
        sys_message = self.progress_task[tag]
        sys_message.set_progress((done_size/total_size) * 100)
        asyncio.run_coroutine_threadsafe(self.socket.send_system_message(sys_message), cast(asyncio.AbstractEventLoop, self.socket.high_priority_loop))
        if sys_message.progress >= 100:
            del self.progress_task[tag]

class ModelSession:
    def __init__(self, model_path: str, title: Optional[str] = None) -> None:
        self.id = uuid.uuid4().hex
        self.mode_path = model_path
        self.date = int(datetime.now().timestamp() * 1000)
        self.filename = f'{self.id}.bin'
        self.title = f'Session-${self.date}' if title is None else title

    def to_json(self) -> dict:
        return {
            'id': self.id,
            'title': self.title,
            'model_path': self.mode_path,
            'date': self.date,
            'filename': self.filename,
        }
    def is_session_valid(self):
        filepath = MODEL_SAVE_PATH / self.filename
        return filepath.exists() and filepath.is_file() and os.path.exists(self.mode_path) and os.path.isfile(self.mode_path)
    
    def get_save_path(self) -> Path:
        return MODEL_SAVE_PATH / self.filename

class FastllamaWebsocket:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.message_manager = MessageManager()
        self.supported_versions = ["1.0"]
        self.file_manager = FileManager()
        self.model_name = 'LLaMa Model'
        self.model: Optional[Model] = None
        self.model_path = '';
        self.current_gen_message: Optional[ModelMessage] = None
        self.prompt_prefix = "\n\n### Instruction:\n\n"
        self.prompt_suffix = "\n\n### Response:\n\n"
        self.high_priority_loop_ready = threading.Event()
        self.high_priority_loop = None
        self.high_priority_thread = Thread(target=self.__run_high_priority_loop)
        self.high_priority_thread.start()
        self.high_priority_loop_ready.wait()
        self.gen_args = {
            "num_tokens": 500,
            "top_p" : 0.95,
            "temp" : 0.8,
            "repeat_penalty" : 1.0,
            "streaming_fn" : self.__stream_token_callback,
            "stop_words" : ["###"],
        }
        self.session_config: List[ModelSession] = []

    def __run_high_priority_loop(self) -> None:
        self.high_priority_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.high_priority_loop)
        self.high_priority_loop_ready.set()
        self.high_priority_loop.run_forever()

    async def __send_message(self, message: dict) -> None:
        if self.websocket.client_state == WebSocketState.CONNECTED:
            await self.websocket.send_json(message)
        else:
            print(f"WebSocket state: {self.websocket.client_state}")

    async def notify_error(self, message: str) -> None:
        print(f"Error: {message}")
        await self.__send_message({
            "message": message,
            "type": "error-notification"
        })
    
    async def notify_info(self, message: str) -> None:
        print(f"Error: {message}")
        await self.__send_message({
            "message": message,
            "type": "info-notification"
        })
    
    async def notify_warning(self, message: str) -> None:
        print(f"Error: {message}")
        await self.__send_message({
            "message": message,
            "type": "warning-notification"
        })

    async def notify_success(self, message: str) -> None:
        print(f"Error: {message}")
        await self.__send_message({
            "message": message,
            "type": "success-notification"
        })

    async def close(self) -> None:
        await self.websocket.close()
        if self.high_priority_loop is not None:
            self.high_priority_loop.call_soon_threadsafe(self.high_priority_loop.stop)
        self.high_priority_thread.join()

    async def __handle_init(self, ws_message: dict) -> None:
        try:
            version = ws_message.get("version")
            if version not in self.supported_versions:
                await self.notify_error(f"Unsupported version: {version}")
                await self.close()
                return
            self.session_config = self.__read_session_config()
            print(f"Session config: {self.session_config}")
            await self.__send_message({
                "type": "init-ack",
                "currentPath": self.file_manager.get_path(),
                "files": self.file_manager.to_json()['files'],
                "saveHistory": [session.to_json() for session in self.session_config],
                "commands": [command.to_json() for command in SUPPORTED_COMMANDS]
            })
        except Exception as e:
            await self.notify_error(str(e))
            await self.close()
            return
        
    async def __handle_user_message(self, ws_message: dict) -> None:
        try:
            if self.model is None:
                await self.notify_error("Model is not loaded")
                return
            (is_valid, err_message) = UserMessage.is_valid(ws_message)
            if "webui_id" not in ws_message:
                await self.notify_error("'webui_id' is required")
                return
            if not is_valid:
                await self.notify_error(err_message)
                return
            status_type = ConversationMessageStatus.from_string(ws_message['status']['kind'])
            message = self.message_manager.make_user_message(ws_message['title'], ws_message['message'], status_type)
            await self.__send_message({
                "type": "message-ack",
                "webui_id": ws_message['webui_id'],
                "status": "success",
                "id": message.id
            })
            
            await self.__generate_message(message)
        except Exception as e:
            await self.notify_error(str(e))
            return
        
    async def send_model_message(self, message: ModelMessage):
        await self.__send_message(message.to_json())

    async def send_system_message(self, message: SystemMessage):
        await self.__send_message(message.to_json())

    async def __handle_file_manager_message(self, ws_message: dict) -> None:
        try:
            (is_valid, err_message) = FileManager.is_valid(ws_message)

            if not is_valid:
                await self.notify_error(err_message)
                return

            if ws_message['kind'] == 'go-back':
                self.file_manager.go_back()
            else:
                self.file_manager.open_dir(path=ws_message['path'])

            json_files = self.file_manager.to_json()
            await self.__send_message({
                "type": "file-manager-ack",
                "currentPath": json_files['path'],
                "files": json_files['files']
            })
        except Exception as e:
            # print(e)
            await self.notify_error(str(e))
            return
        
    async def __handle_model_init(self, ws_message: dict) -> None:
        try:
            if 'model_path' not in ws_message:
                await self.notify_error("'model_path' is required")
                return
            
            if type(ws_message['model_path']) != str:
                await self.notify_error("'model_path' must be a string")
                return
            
            if not os.path.isfile(ws_message['model_path']):
                await self.notify_error(f"Model file '{ws_message['model_path']}' does not exist")
                return
            
            n_threads = ws_message.get('n_threads', 4)
            n_ctx = ws_message.get('n_ctx', 512)
            last_n_size = ws_message.get('last_n_size', 64)
            n_batch = ws_message.get('n_batch', 128)
            load_parallel = ws_message.get('load_parallel', False)
            seed = ws_message.get('seed', 0)
            tokens_to_keep = ws_message.get('tokens_to_keep', 200)
            use_mmap = ws_message.get('use_mmap', False)
            use_mlock = ws_message.get('use_mlock', False)
            n_load_parallel_blocks = ws_message.get('n_load_parallel_blocks', 1)

            self.model_path = ws_message['model_path']
            self.model = Model(
                path=self.model_path,
                num_threads=n_threads,
                n_ctx=n_ctx,
                last_n_size=last_n_size,
                n_batch=n_batch,
                load_parallel=load_parallel,
                seed=seed,
                tokens_to_keep=tokens_to_keep,
                use_mmap=use_mmap,
                use_mlock=use_mlock,
                n_load_parallel_blocks=n_load_parallel_blocks,
                logger=FastllamaWebsocketLogger(self)
            )
        except Exception as e:
            await self.notify_error(str(e))
            return
    def __stream_token_callback(self, message: str):
        if self.current_gen_message is None:
            self.current_gen_message = self.message_manager.make_model_message(self.model_name, "", ConversationMessageStatus.LOADING)
        self.current_gen_message.message += message
        asyncio.run_coroutine_threadsafe(self.send_model_message(self.current_gen_message), cast(asyncio.AbstractEventLoop, self.high_priority_loop))

    async def __generate_message(self, user_message: UserMessage) -> None:
        try:
            if self.model is None:
                await self.notify_error("Model is not loaded")
                return
            prompt = f"{self.prompt_prefix}{user_message.message}{self.prompt_suffix}"
            self.model.ingest(prompt)
            user_message.status = ConversationMessageStatus.SUCCESS
            await self.__send_message(user_message.to_json())
            
            self.model.generate(**self.gen_args)
            if self.current_gen_message is not None:
                self.current_gen_message.status = ConversationMessageStatus.SUCCESS
                await self.__send_message(self.current_gen_message.to_json())
            self.current_gen_message = None
        except Exception as e:
            await self.notify_error(str(e))
            return

    def __handle_invoke_commands__set(self, args: list) -> None:
        for arg in args:
            cmd_arg = CommandArg.from_json(arg)

            if arg['name'] == 'p_prefix':
                self.prompt_prefix = cmd_arg.as_normalized_value()
            elif arg['name'] == 'p_suffix':
                self.prompt_suffix = cmd_arg.as_normalized_value()
            elif arg['name'] == 'max_gen_token_length':
                self.gen_args['num_tokens'] = cmd_arg.as_normalized_value()
            elif arg['name'] == 'top_k':
                self.gen_args['top_k'] = cmd_arg.as_normalized_value()
            elif arg['name'] == 'top_p':
                self.gen_args['top_p'] = cmd_arg.as_normalized_value()
            elif arg['name'] == 'temp':
                self.gen_args['temp'] = cmd_arg.as_normalized_value()

    async def __handle_invoke_commands__reset(self, args: list) -> None:
        if self.model is None:
            return
        
        self.model.reset()

    async def __handle_invoke_commands(self, ws_message: dict) -> None:
        try:
            if self.model is None:
                await self.notify_error("Model is not loaded")
                return
            if 'command' not in ws_message:
                await self.notify_error("'command' is required")
                return
            
            if type(ws_message['command']) != str:
                await self.notify_error("'command' must be a string")
                return
            
            if 'args' in ws_message and type(ws_message['args']) != list:
                await self.notify_error("'args' must be a list")
                return
            
            if ws_message['command'] == 'set':
                await self.__handle_invoke_commands__set(ws_message['args'])
                await self.notify_success("Command 'set' executed successfully")
            elif ws_message['command'] == 'reset':
                await self.__handle_invoke_commands__reset(ws_message['args'])
                await self.notify_success("Command 'reset' executed successfully")
        except Exception as e:
            await self.notify_error(str(e))
            return

    def __read_session_config(self) -> List[ModelSession]:
        path = MODEL_SAVE_PATH / "session-data.json"
        if not path.exists():
            return []
        res = [];
        try:
            with open(path, 'r') as f:
                temp = json.load(f)
                for s in temp:
                    print(s)
                    session = ModelSession(model_path=s['model_path'])
                    session.id = s['id']
                    session.date=s['date']
                    session.filename=s['filename']
                    if session.is_session_valid():
                        res.append(session)
        except Exception as e:
            print(f"Failed to read session config: {e}")
        return res
        
    def __write_session_config(self) -> None:
        path = MODEL_SAVE_PATH / "session-data.json"
        if not path.parent.exists():
            path.parent.mkdir()
        with open(path, 'w') as f:
            json.dump([m.to_json() for m in self.session_config], f)

    async def __send_session_list(self) -> None:
        await self.__send_message({
            'type': 'session-list-ack',
            'sessions': [s.to_json() for s in self.session_config]
        })

    async def __handle_model_session_save(self, _ws_message: dict) -> None:
        try:
            if self.model is None:
                await self.notify_error("Model is not loaded")
                return

            message_text: Optional[str] = self.message_manager.get_save_title(size=40)
            
            session = ModelSession(model_path=self.model_path, title=message_text)
            if not MODEL_SAVE_PATH.exists():
                MODEL_SAVE_PATH.mkdir()

            if self.model.save_state(str(session.get_save_path())) == True:
                self.session_config.append(session)
                self.__write_session_config()
                
                await self.__send_session_list()
            else:
                await self.notify_error("Failed to save session")
        except Exception as e:
            await self.notify_error(str(e))
            return
        
    async def __handle_model_session_delete(self, ws_message: dict) -> None:
        try:
            if 'id' not in ws_message:
                await self.notify_error("'id' is required")
                return
            
            if type(ws_message['id']) != str:
                await self.notify_error("'id' must be a string")
                return
            index = -1
            for i, s in enumerate(self.session_config):
                if s.id == ws_message['id']:
                    index = i
                    break
            if index == -1:
                await self.notify_error("Session not found")
                return
            path = self.session_config[index].get_save_path()
            if path.exists():
                path.unlink()
            
            self.session_config.remove(self.session_config[index])
            self.__write_session_config()
            await self.__send_session_list()
        except Exception as e:
            print(e)
            await self.notify_error(str(e))
            return
        
    async def __handle_model_session_load(self, ws_message: dict) -> None:
        try:
            if self.model is None:
                await self.notify_error("Model is not loaded")
                return
            
            if 'id' not in ws_message:
                await self.notify_error("'id' is required")
                return

            if type(ws_message['id']) != str:
                await self.notify_error("'id' must be a string")
                return
            
            index = -1
            for i, s in enumerate(self.session_config):
                if s.id == ws_message['id']:
                    index = i
                    break
            if index == -1:
                await self.notify_error("Session not found")
                return
            
            session = self.session_config[index]
            if not session.is_session_valid():
                await self.notify_error("Session is not valid")
                return
            if os.path.realpath(session.mode_path) != os.path.realpath(self.model_path):
                await self.notify_error("Model session is not compatible with current model")
                return
            
            if not self.model.load_state(str(session.get_save_path())):
                await self.notify_error("Failed to load session")
                return
        except Exception as e:
            print(e)
            await self.notify_error(str(e))
            return

    async def handle_ws_message(self, ws_message: dict) -> None:
        message_type = ws_message.get("type")
        print(ws_message)
        if message_type == 'init':
            await self.__handle_init(ws_message)
        elif message_type == 'init-model':
            await self.__handle_model_init(ws_message)
        elif message_type == 'user-message':
            await self.__handle_user_message(ws_message)
        elif message_type == 'file-manager':
            await self.__handle_file_manager_message(ws_message)
        elif message_type == 'invoke-command':
            await self.__handle_invoke_commands(ws_message)
        elif message_type == 'session-save':
            await self.__handle_model_session_save(ws_message)
        elif message_type == 'session-delete':
            await self.__handle_model_session_delete(ws_message)
        elif message_type == 'session-load':
            await self.__handle_model_session_load(ws_message)
        
            
            
