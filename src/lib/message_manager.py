from enum import Enum
from typing import Any, Dict, List, MutableMapping, Optional, Union, cast
import uuid


class MessageKind(Enum):
    USER = 1
    SYSTEM = 2
    MODEL = 3

    @staticmethod
    def from_string(kind: str) -> 'MessageKind':
        if kind == 'user':
            return MessageKind.USER
        elif kind == 'system':
            return MessageKind.SYSTEM
        elif kind == 'model':
            return MessageKind.MODEL
        else:
            raise ValueError('Invalid message kind')
        
    def to_string(self) -> str:
        if self == MessageKind.USER:
            return 'user'
        elif self == MessageKind.SYSTEM:
            return 'system'
        elif self == MessageKind.MODEL:
            return 'model'
        else:
            raise ValueError('Invalid message kind')

class SystemMessageKind(Enum):
    INFO = 1
    WARNING = 2
    ERROR = 3
    PROGRESS = 4

    @staticmethod
    def from_string(kind: str) -> 'SystemMessageKind':
        if kind == 'info':
            return SystemMessageKind.INFO
        elif kind == 'warning':
            return SystemMessageKind.WARNING
        elif kind == 'error':
            return SystemMessageKind.ERROR
        elif kind == 'progress':
            return SystemMessageKind.PROGRESS
        else:
            raise ValueError('Invalid system message kind')
        
    def to_string(self) -> str:
        if self == SystemMessageKind.INFO:
            return 'info'
        elif self == SystemMessageKind.WARNING:
            return 'warning'
        elif self == SystemMessageKind.ERROR:
            return 'error'
        elif self == SystemMessageKind.PROGRESS:
            return 'progress'
        else:
            raise ValueError('Invalid system message kind')

class ConversationMessageStatus(Enum):
    LOADING = 1
    PROGRESS = 2
    SUCCESS = 3
    FAILURE = 4

    @staticmethod
    def from_string(status: str) -> 'ConversationMessageStatus':
        if status == 'loading':
            return ConversationMessageStatus.LOADING
        elif status == 'progress':
            return ConversationMessageStatus.PROGRESS
        elif status == 'success':
            return ConversationMessageStatus.SUCCESS
        elif status == 'failure':
            return ConversationMessageStatus.FAILURE
        else:
            raise ValueError('Invalid conversation message status')
        
    def to_string(self) -> str:
        if self == ConversationMessageStatus.LOADING:
            return 'loading'
        elif self == ConversationMessageStatus.PROGRESS:
            return 'progress'
        elif self == ConversationMessageStatus.SUCCESS:
            return 'success'
        elif self == ConversationMessageStatus.FAILURE:
            return 'failure'
        else:
            raise ValueError('Invalid conversation message status')

class Message:
    def __init__(self, kind: MessageKind, id = uuid.uuid4()) -> None:
        self.id = str(id)
        self.message_type = kind

    def as_system_message(self) -> 'SystemMessage':
        return cast(SystemMessage, self)
    
    def as_user_message(self) -> 'UserMessage':
        return cast(UserMessage, self)
    
    def as_model_message(self) -> 'ModelMessage':
        return cast(ModelMessage, self)
    
    def is_system_message(self) -> bool:
        return self.message_type == MessageKind.SYSTEM
    
    def is_user_message(self) -> bool:
        return self.message_type == MessageKind.USER
    
    def is_model_message(self) -> bool:
        return self.message_type == MessageKind.MODEL
    
    def to_json(self) -> dict:
        return {
            'id': self.id,
            'type': self.message_type.to_string(),
        }

class SystemMessage(Message):
    def __init__(self, kind: SystemMessageKind, function: str, message: str):
        super().__init__(MessageKind.SYSTEM)
        self.kind = kind
        self.function = function
        self.message = message
        self.progress = 0.0

    def is_info(self) -> bool:
        return self.kind == SystemMessageKind.INFO
    
    def is_warning(self) -> bool:
        return self.kind == SystemMessageKind.WARNING
    
    def is_error(self) -> bool:
        return self.kind == SystemMessageKind.ERROR
    
    def is_progress(self) -> bool:
        return self.kind == SystemMessageKind.PROGRESS
    
    def set_progress(self, progress: float) -> None:
        self.progress = progress

    def to_json(self) -> dict:
        temp: Dict[str, Union[str, float]] = {
            'id': self.id,
            'type': self.message_type.to_string(),
            'kind': self.kind.to_string(),
            'function': self.function,
            'message': self.message,
        }
        if self.is_progress():
            temp['progress'] = self.progress
        return temp


class UserMessage(Message):
    def __init__(self, title: str, message: str, status: ConversationMessageStatus = ConversationMessageStatus.SUCCESS):
        super().__init__(MessageKind.USER)
        self.message = message
        self.title = title
        self.status = status
        self.progress = 0.0

    def is_loading(self) -> bool:
        return self.status == ConversationMessageStatus.LOADING
    
    def is_progress(self) -> bool:
        return self.status == ConversationMessageStatus.PROGRESS
    
    def is_success(self) -> bool:
        return self.status == ConversationMessageStatus.SUCCESS
    
    def is_failure(self) -> bool:
        return self.status == ConversationMessageStatus.FAILURE
    
    def set_progress(self, progress: float) -> None:
        self.progress = progress

    def to_json(self) -> dict:
        temp = {
            'id': self.id,
            'type': self.message_type.to_string(),
            'title': self.title,
            'message': self.message,
            'status': {},
        }
        status: MutableMapping[str, Union[str, float]] = {
            'type': self.status.to_string(),
        }

        if self.is_progress():
            status['progress'] = self.progress

        temp['status'] = status
        return temp

class ModelMessage(UserMessage):
    def __init__(self, title: str, message: str, status: ConversationMessageStatus = ConversationMessageStatus.SUCCESS):
        super().__init__(title, message, status)
        self.message_type = MessageKind.MODEL
        self.message = message

class MessageManager:
    def __init__(self) -> None:
        self.messages: List[Message] = []

    def add_message(self, message: Message) -> None:
        self.messages.append(message)

    def make_system_message(self, kind: SystemMessageKind, function: str, message: str) -> SystemMessage:
        system_message = SystemMessage(kind, function, message)
        self.add_message(system_message)
        return system_message

    def make_user_message(self, title: str, message: str, status: ConversationMessageStatus = ConversationMessageStatus.SUCCESS, id: Optional[str] = None) -> UserMessage:
        user_message = UserMessage(title, message, status)
        if (id is not None):
            user_message.id = id
        self.add_message(user_message)
        return user_message
    
    def make_model_message(self, title: str, message: str, status: ConversationMessageStatus = ConversationMessageStatus.SUCCESS) -> ModelMessage:
        model_message = ModelMessage(title, message, status)
        self.add_message(model_message)
        return model_message
    
    def get_messages(self) -> List[Message]:
        return self.messages
    
    def get_last_message(self) -> Message:
        return self.messages[-1]
    
    def get_message(self, id: str) -> Optional[Message]:
        for message in self.messages:
            if message.id == id:
                return message
        return None
    
    def remove_message(self, id: str) -> None:
        for message in self.messages:
            if message.id == id:
                self.messages.remove(message)
                return
        return None
    
    def to_json(self) -> dict:
        return {
            'messages': [message.to_json() for message in self.messages]
        }