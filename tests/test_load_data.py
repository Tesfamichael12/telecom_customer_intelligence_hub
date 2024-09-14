# tests/test_data_loader.py

import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.load_data import DataLoader

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.data_loader = DataLoader()

    @patch('src.data_loader.create_engine')
    def test_load_data(self, mock_create_engine):
        # Mock the database connection and query execution
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        mock_engine.execute.return_value = mock_df

        result = self.data_loader.load_data("SELECT * FROM test_table")
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertEqual(list(result.columns), ['col1', 'col2'])

    def test_clean_data(self):
        # Create a sample DataFrame with some issues to clean
        df = pd.DataFrame({
            'MSISDN/Number': ['1', '2', '3', '4', '5'],
            'Handset Type': ['A', 'B', 'undefined', 'D', 'E'],
            'Handset Manufacturer': ['X', 'Y', 'Z', 'undefined', 'W'],
            'Dur. (ms)': [100, 200, 300, 400, 500],
            'Total DL (Bytes)': [1000, 2000, 3000, 4000, 5000],
            'Total UL (Bytes)': [100, 200, 300, 400, 500]
        })

        cleaned_df = self.data_loader.clean_data(df)

        # Check if rows with 'undefined' values are removed
        self.assertEqual(len(cleaned_df), 3)
        
        # Check if all required columns are present
        required_columns = ['MSISDN/Number', 'Handset Type', 'Handset Manufacturer', 'Dur. (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)']
        self.assertTrue(all(col in cleaned_df.columns for col in required_columns))

if __name__ == '__main__':
    unittest.main()