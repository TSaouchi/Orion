import unittest
from unittest.mock import patch, mock_open, MagicMock
import numpy as np
import pandas as pd
import os

from Core.Reader import Reader
from DataProcessor import *

class TestReader(unittest.TestCase):

    def setUp(self):
        # Setup paths and patterns for the Reader instance
        self.path = r"C:\toto"
        self.patterns = ["toto"]
        self.reader = Reader(self.path, self.patterns)

    @patch('builtins.open', new_callable=mock_open, read_data="Header Line\nColumn1,Column2\n1,2\n3,4")
    @patch('os.path.join', return_value="mocked_file_path")
    @patch.object(Reader, '_Reader__generate_file_list')
    @patch.object(Reader, '_Reader__file_not_found')
    def test_read_ascii_single_file(self, mock_file_not_found, mock_generate_file_list, mock_path_join, mock_open):
        # Mock __generate_file_list to return a single file
        mock_generate_file_list.return_value = {'file1': 'mocked_file'}

        # Call the method under test
        result_base = self.reader.read_ascii()

        # Check if the base is correctly initialized with one instant
        self.assertEqual(len(result_base.keys()), 1)
        self.assertIn('Column1', list(result_base[0][0].keys())[0])
        self.assertIn('Column2', list(result_base[0][0].keys())[1])

        # Verify file_not_found was called
        mock_file_not_found.assert_called_once()

    @patch('builtins.open', new_callable=mock_open, read_data="Header Line\nColumn1,Column2\n1,2\n3,4")
    @patch('os.path.join', return_value="mocked_file_path")
    @patch.object(Reader, '_Reader__generate_file_list')
    @patch.object(Reader, '_Reader__file_not_found')
    def test_read_ascii_multiple_files(self, mock_file_not_found, mock_generate_file_list, mock_path_join, mock_open):
        # Mock __generate_file_list to return multiple files
        mock_generate_file_list.return_value = {'file1': 1, 'file2': 2}

        # Call the method under test
        result_base = self.reader.read_ascii()

        # Check if the base is correctly initialized with multiple instants
        self.assertEqual(len(result_base[0].keys()), 2)
        self.assertIn('Column1', list(result_base[0][0].keys())[0])
        self.assertIn('Column2', list(result_base[0][0].keys())[1])

        # Verify file_not_found was called
        mock_file_not_found.assert_called_once()

    @patch('builtins.open', new_callable=mock_open, read_data="Header Line\nColumn1,Column2\n1,2\n3,4")
    @patch('os.path.join', return_value="mocked_file_path")
    @patch.object(Reader, '_Reader__generate_file_list')
    @patch.object(Reader, '_Reader__file_not_found')
    def test_read_ascii_with_file_selection(self, mock_file_not_found, mock_generate_file_list, mock_path_join, mock_open):
        # Mock __generate_file_list to return multiple files
        mock_generate_file_list.return_value = {'file1': 1, 'file2': 2, 'file3': 3}

        # Select files 2 and 3
        file_selection = [2, 3]

        # Call the method under test with file selection
        result_base = self.reader.read_ascii(file_selection=file_selection)

        # Check if the correct files were read
        self.assertEqual(len(result_base[0].keys()), 2)
        self.assertIn('Column1', list(result_base[0][0].keys())[0])
        self.assertIn('Column2', list(result_base[0][0].keys())[1])

        # Verify file_not_found was called
        mock_file_not_found.assert_called_once()

if __name__ == '__main__':
    unittest.main()
