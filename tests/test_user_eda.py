# tests/test_user_overview.py

import unittest
import pandas as pd
import numpy as np
from src.user_eda import UserOverview

class TestUserOverview(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        self.df = pd.DataFrame({
            'MSISDN/Number': ['1', '2', '3', '4', '5'],
            'Handset Type': ['A', 'B', 'C', 'D', 'E'],
            'Handset Manufacturer': ['X', 'Y', 'Z', 'W', 'V'],
            'Dur. (ms)': [100, 200, 300, 400, 500],
            'Total DL (Bytes)': [1000, 2000, 3000, 4000, 5000],
            'Total UL (Bytes)': [100, 200, 300, 400, 500]
        })
        self.user_overview = UserOverview(self.df)

    def test_describe_dataset(self):
        result = self.user_overview.describe_dataset()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue('mean' in result.index)
        self.assertTrue('std' in result.index)

    def test_segment_and_compute_decile(self):
        result = self.user_overview.segment_and_compute_decile()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)  # Check if it returns top 5 deciles
        self.assertTrue('Total Data (Bytes)' in result.columns)

    def test_univariate_analysis(self):
        result = self.user_overview.univariate_analysis()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue('Mean' in result.columns)
        self.assertTrue('Median' in result.columns)
        self.assertTrue('Variance' in result.columns)

    def test_bivariate_analysis(self):
        # Add necessary columns for bivariate analysis
        self.df['Social Media DL (Bytes)'] = np.random.randint(1, 1000, 5)
        self.df['Social Media UL (Bytes)'] = np.random.randint(1, 1000, 5)
        
        result = self.user_overview.bivariate_analysis()
        self.assertIsInstance(result, dict)
        self.assertTrue('correlation_matrix' in result)
        self.assertIsInstance(result['correlation_matrix'], pd.DataFrame)

    def test_pca_analysis(self):
        result = self.user_overview.pca_analysis()
        self.assertIsInstance(result, dict)
        self.assertTrue('explained_variance' in result)
        self.assertTrue('loadings' in result)
        self.assertIsInstance(result['loadings'], pd.DataFrame)

if __name__ == '__main__':
    unittest.main()