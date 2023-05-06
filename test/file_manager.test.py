from pathlib import Path
import unittest
import os
import sys

CURRENT_FILE_PATH = Path(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(CURRENT_FILE_PATH, '..', 'src'))

from utils.file_manager import FileManager, FileStructure, FileKind, workspace_path

class TestFileManager(unittest.TestCase):
    def test_workspace(self):
        workspace_abs_path = os.path.abspath(workspace_path())
        assert workspace_abs_path == os.path.abspath(os.path.join(CURRENT_FILE_PATH, '..'))

    def test_file_structure(self):
        file_structure = FileStructure(workspace_path(), 'file_manager.test.py', FileKind.FILE)
        self.assertEqual(file_structure.get_path(), workspace_path())
        self.assertEqual(file_structure.get_file_name(), 'file_manager.test.py')
        self.assertEqual(file_structure.get_file(), workspace_path() / 'file_manager.test.py')
        self.assertTrue(file_structure.is_file())
        self.assertFalse(file_structure.is_dir())
        json = file_structure.to_json()
        self.assertEqual(json['path'], str(workspace_path() / 'file_manager.test.py'))
        self.assertEqual(json['file_name'], 'file_manager.test.py')
        self.assertEqual(json['kind'], 'file')

    def test_file_manager(self):
        dir_path = CURRENT_FILE_PATH / 'file_manager_test_dir'
        file_manager = FileManager(dir_path)
        self.assertEqual(file_manager.get_path(), dir_path)
        file_list = file_manager.get_file_list()
        self.assertEqual(len(file_list), 2)
        files_found = 0
        for file in file_list:
            if file.get_file_name() == 'images':
                self.assertTrue(file.is_dir())
                self.assertFalse(file.is_file())
                self.assertEqual(file.get_file(), dir_path / 'images')
                files_found += 1
            if file.get_file_name() == 'test.txt':
                self.assertTrue(file.is_file())
                self.assertFalse(file.is_dir())
                self.assertEqual(file.get_file(), dir_path / 'test.txt')
                files_found += 1

        self.assertEqual(files_found, 2)

    def test_file_manager_go_back(self):
        dir_path = CURRENT_FILE_PATH / 'file_manager_test_dir'
        file_manager = FileManager(dir_path)
        self.assertEqual(file_manager.get_path(), dir_path)
        file_manager.go_back()
        self.assertEqual(file_manager.get_path(), CURRENT_FILE_PATH)

    def test_open_dir(self):
        dir_path = CURRENT_FILE_PATH / 'file_manager_test_dir'
        file_manager = FileManager(dir_path)
        self.assertEqual(file_manager.get_path(), dir_path)
        file_manager.open_dir('images')
        self.assertEqual(file_manager.get_path(), dir_path / 'images')
        self.assertEqual(len(file_manager.get_file_list()), 1)

    def test_file_manager_to_json(self):
        dir_path = CURRENT_FILE_PATH / 'file_manager_test_dir'
        file_manager = FileManager(dir_path)
        json = file_manager.to_json()
        self.assertEqual(json['path'], str(dir_path))
        self.assertEqual(len(json['files']), 2)
        self.assertEqual(json['files'][0]['path'], str(dir_path / 'images'))
        self.assertEqual(json['files'][0]['file_name'], 'images')
        self.assertEqual(json['files'][0]['kind'], 'directory')
        self.assertEqual(json['files'][1]['path'], str(dir_path / 'test.txt'))
        self.assertEqual(json['files'][1]['file_name'], 'test.txt')
        self.assertEqual(json['files'][1]['kind'], 'file')
    

if __name__ == '__main__':
    unittest.main()