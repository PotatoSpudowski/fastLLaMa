from pathlib import Path
import unittest
import os
import sys

CURRENT_FILE_PATH = Path(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(CURRENT_FILE_PATH, '..', 'src'))

from utils.message_manager import *

class TestCommand(unittest.TestCase):
    def test_message(self):
        message = Message(MessageKind.USER)
        self.assertEqual(message.message_type, MessageKind.USER)
        self.assertEqual(message.to_json(), {
            'id': message.id,
            'type': 'user',
        })

    def test_system_message(self):
        message = SystemMessage(SystemMessageKind.INFO, 'test', 'test')
        self.assertEqual(message.message_type, MessageKind.SYSTEM)
        self.assertEqual(message.kind, SystemMessageKind.INFO)
        self.assertEqual(message.function, 'test')
        self.assertEqual(message.message, 'test')
        self.assertEqual(message.progress, 0.0)
        self.assertEqual(message.to_json(), {
            'id': message.id,
            'type': 'system',
            'kind': 'info',
            'function': 'test',
            'message': 'test',
        })

    def test_user_message(self):
        message = UserMessage('test', 'This is a test message.', ConversationMessageStatus.LOADING)
        self.assertEqual(message.message_type, MessageKind.USER)
        self.assertEqual(message.message, 'This is a test message.')
        self.assertEqual(message.status, ConversationMessageStatus.LOADING)
        self.assertEqual(message.to_json(), {
            'id': message.id,
            'type': 'user',
            'title': 'test',
            'message': 'This is a test message.',
            'status': {
                'type': 'loading',
            },
        })

    def test_model_message(self):
        message = ModelMessage('test', 'test', ConversationMessageStatus.PROGRESS)
        self.assertEqual(message.message_type, MessageKind.MODEL)
        self.assertEqual(message.status, ConversationMessageStatus.PROGRESS)
        self.assertEqual(message.title, 'test')
        self.assertEqual(message.message, 'test')
        self.assertEqual(message.progress, 0.0)
        self.assertEqual(message.to_json(), {
            'id': message.id,
            'type': 'model',
            'title': 'test',
            'message': 'test',
            'status': {
                'progress': 0.0,
                'type': 'progress'
            }
        })

    def test_message_manager(self):
        manager = MessageManager()
        self.assertEqual(manager.messages, [])
        self.assertEqual(manager.to_json(), {
            'messages': [],
        })

        manager.add_message(UserMessage('test', 'This is a test message.', ConversationMessageStatus.LOADING))
        self.assertEqual(len(manager.messages), 1)
        self.assertEqual(manager.to_json(), {
            'messages': [{
                'id': manager.messages[0].id,
                'type': 'user',
                'title': 'test',
                'message': 'This is a test message.',
                'status': {
                    'type': 'loading',
                },
            }],
        })

        manager.add_message(SystemMessage(SystemMessageKind.INFO, 'test', 'test'))
        self.assertEqual(len(manager.messages), 2)
        self.assertEqual(manager.to_json(), {
            'messages': [{
                'id': manager.messages[0].id,
                'type': 'user',
                'title': 'test',
                'message': 'This is a test message.',
                'status': {
                    'type': 'loading',
                },
            }, {
                'id': manager.messages[1].id,
                'type': 'system',
                'kind': 'info',
                'function': 'test',
                'message': 'test',
            }],
        })

        manager.add_message(ModelMessage('test', 'test', ConversationMessageStatus.PROGRESS))
        self.assertEqual(len(manager.messages), 3)
        self.assertEqual(manager.to_json(), {
            'messages': [{
                'id': manager.messages[0].id,
                'type': 'user',
                'title': 'test',
                'message': 'This is a test message.',
                'status': {
                    'type': 'loading',
                },
            }, {
                'id': manager.messages[1].id,
                'type': 'system',
                'kind': 'info',
                'function': 'test',
                'message': 'test',
            }, {
                'id': manager.messages[2].id,
                'type': 'model',
                'title': 'test',
                'message': 'test',
                'status': {
                    'progress': 0.0,
                    'type': 'progress'
                }
            }],
        })

if __name__ == '__main__':
    unittest.main()