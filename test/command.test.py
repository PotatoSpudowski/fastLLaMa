from pathlib import Path
import unittest
import os
import sys

CURRENT_FILE_PATH = Path(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(CURRENT_FILE_PATH, '..', 'src'))

from lib.command import Command, CommandArgType, CommandArg

class TestCommand(unittest.TestCase):
    
    def test_command_arg(self):
        arg = CommandArg('test', CommandArgType.STRING, 'test')
        self.assertEqual(arg.get_name(), 'test')
        self.assertEqual(arg.get_type(), CommandArgType.STRING)
        self.assertEqual(arg.get_value(), 'test')
        self.assertTrue(arg.is_string())
        self.assertFalse(arg.is_int())
        self.assertFalse(arg.is_float())
        self.assertFalse(arg.is_boolean())
        self.assertEqual(arg.as_string(), 'test')
        self.assertRaises(ValueError, arg.as_int)
        self.assertRaises(ValueError, arg.as_float)
        self.assertTrue(arg.as_boolean())
        self.assertFalse(arg.is_none())

    def test_command(self):
        cmd = Command('test', [
            CommandArg('test', CommandArgType.STRING, 'test'),
            CommandArg('test2', CommandArgType.INT, 1),
            CommandArg('test3', CommandArgType.FLOAT, 1.0),
            CommandArg('test4', CommandArgType.BOOLEAN, True),
        ])
        self.assertEqual(cmd.get_name(), 'test')
        self.assertEqual(len(cmd.args), 4)
        self.assertEqual(cmd.args['test'].get_name(), 'test')
        self.assertEqual(cmd.args['test'].get_type(), CommandArgType.STRING)
        self.assertEqual(cmd.args['test'].get_value(), 'test')
        self.assertTrue(cmd.args['test'].is_string())
        self.assertFalse(cmd.args['test'].is_int())
        self.assertFalse(cmd.args['test'].is_float())
        self.assertFalse(cmd.args['test'].is_boolean())
        self.assertEqual(cmd.args['test'].as_string(), 'test')
        self.assertRaises(ValueError, cmd.args['test'].as_int)
        self.assertRaises(ValueError, cmd.args['test'].as_float)
        self.assertTrue(cmd.args['test'].as_boolean())
        self.assertFalse(cmd.args['test'].is_none())

        self.assertEqual(cmd.args['test2'].get_name(), 'test2')
        self.assertEqual(cmd.args['test2'].get_type(), CommandArgType.INT)
        self.assertEqual(cmd.args['test2'].get_value(), 1)
        self.assertFalse(cmd.args['test2'].is_string())
        self.assertTrue(cmd.args['test2'].is_int())
        self.assertFalse(cmd.args['test2'].is_float())
        self.assertFalse(cmd.args['test2'].is_boolean())
        self.assertEqual(cmd.args['test2'].as_int(), 1)
        self.assertEqual(cmd.args['test2'].as_float(), 1.0)
        self.assertTrue(cmd.args['test2'].as_boolean())
        self.assertFalse(cmd.args['test2'].is_none())

        self.assertEqual(cmd.args['test3'].get_name(), 'test3')
        self.assertEqual(cmd.args['test3'].get_type(), CommandArgType.FLOAT)
        self.assertEqual(cmd.args['test3'].get_value(), 1.0)
        self.assertFalse(cmd.args['test3'].is_string())
        self.assertFalse(cmd.args['test3'].is_int())
        self.assertTrue(cmd.args['test3'].is_float())
        self.assertFalse(cmd.args['test3'].is_boolean())
        self.assertEqual(cmd.args['test3'].as_int(), 1)
        self.assertEqual(cmd.args['test3'].as_float(), 1.0)
        self.assertTrue(cmd.args['test3'].as_boolean())
        self.assertFalse(cmd.args['test3'].is_none())

        self.assertEqual(cmd.args['test4'].get_name(), 'test4')
        self.assertEqual(cmd.args['test4'].get_type(), CommandArgType.BOOLEAN)
        self.assertEqual(cmd.args['test4'].get_value(), True)
        self.assertFalse(cmd.args['test4'].is_string())
        self.assertFalse(cmd.args['test4'].is_int())
        self.assertFalse(cmd.args['test4'].is_float())
        self.assertTrue(cmd.args['test4'].is_boolean())
        self.assertEqual(cmd.args['test4'].as_int(), 1)
        self.assertEqual(cmd.args['test4'].as_float(), 1.0)
        self.assertTrue(cmd.args['test4'].as_boolean())
        self.assertFalse(cmd.args['test4'].is_none())

    def test_normalized_value(self):
        int_arg = CommandArg('test', CommandArgType.INT, '12')
        self.assertEqual(int_arg.as_normalized_value(), 12)

        float_arg = CommandArg('test', CommandArgType.FLOAT, '12.0')
        self.assertEqual(float_arg.as_normalized_value(), 12.0)

        boolean_arg = CommandArg('test', CommandArgType.BOOLEAN, '')
        self.assertEqual(boolean_arg.as_normalized_value(), False)

        invalid_int_arg = CommandArg('test', CommandArgType.INT, 'test')
        self.assertIsNone(invalid_int_arg.as_normalized_value())

        invalid_float_arg = CommandArg('test', CommandArgType.FLOAT, 'test')
        self.assertIsNone(invalid_float_arg.as_normalized_value())

if __name__ == '__main__':
    unittest.main()