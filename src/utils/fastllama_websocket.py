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

MODEL_PATH = "../models/VICUNA-7B/ggml-vicuna-7b-1.0-uncensored-q4_2.bin"

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



class FastllamaWebsocket:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.message_manager = MessageManager()
        self.supported_versions = ["1.0"]
        self.file_manager = FileManager()
        self.model_name = 'LLaMa Model'
        self.model: Optional[Model] = None
        self.current_gen_message: Optional[ModelMessage] = None
        self.prompt_prefix = "\n\n### Instruction:\n\n"
        self.prompt_suffix = "\n\n### Response:\n\n"
        self.high_priority_loop_ready = threading.Event()
        self.high_priority_loop = None
        self.high_priority_thread = Thread(target=self.__run_high_priority_loop)
        self.high_priority_thread.start()
        self.high_priority_loop_ready.wait()

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
            "type": "error"
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
            await self.__send_message({
                "type": "init-ack",
                "currentPath": self.file_manager.get_path(),
                "files": self.file_manager.to_json()['files'],
                "saveHistory": [],
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

            model_path = ws_message['model_path']
            self.model = Model(
                path=model_path,
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
            
            self.model.generate(
                num_tokens=500,
                top_p=0.95,
                temp=0.8,
                repeat_penalty=1.0,
                streaming_fn=self.__stream_token_callback,
                stop_words=["###"],
            )
            if self.current_gen_message is not None:
                self.current_gen_message.status = ConversationMessageStatus.SUCCESS
                await self.__send_message(self.current_gen_message.to_json())
            self.current_gen_message = None
        except Exception as e:
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
        
            
            
