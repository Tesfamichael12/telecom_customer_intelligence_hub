# tests/test_user_engagement.py

import unittest
import pandas as pd
import numpy as np
from src.user_engagement_analysis import UserEngagement

class TestUserEngagement(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        self.df = pd.DataFrame({
            'MSISDN/Number': ['1', '2', '3', '4', '5'] * 20,
            'Dur. (ms)': np.random.randint(100, 1000, 100),
            'Total DL (Bytes)': np.random.randint(1000, 10000, 100),
            'Total UL (Bytes)': np.random.randint(100, 1000, 100),
            'Social Media DL (Bytes)': np.random.randint(100, 1000, 100),
            'Social Media UL (Bytes)': np.random.randint(10, 100, 100),
            'Google DL (Bytes)': np.random.randint(100, 1000, 100),
            'Google UL (Bytes)': np.random.randint(10, 100, 100),
            'Email DL (Bytes)': np.random.randint(100, 1000, 100),
            'Email UL (Bytes)': np.random.randint(10, 100, 100),
            'Youtube DL (Bytes)': np.random.randint(100, 1000, 100),
            'Youtube UL (Bytes)': np.random.randint(10, 100, 100),
            'Netflix DL (Bytes)': np.random.randint(100, 1000, 100),
            'Netflix UL (Bytes)': np.random.randint(10, 100, 100),
            'Gaming DL (Bytes)': np.random.randint(100, 1000, 100),
            'Gaming UL (Bytes)': np.random.randint(10, 100, 100),
            'Other DL (Bytes)': np.random.randint(100, 1000, 100),
            'Other UL (Bytes)': np.random.randint(10, 100, 100)
        })
        self.user_engagement = UserEngagement(self.df)

    def test_aggregate_user_metrics(self):
        result = self.user_engagement.aggregate_user_metrics()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)  # 5 unique users
        self.assertTrue('Sessions Frequency' in result.columns)
        self.assertTrue('Total Session Duration (ms)' in result.columns)
        self.assertTrue('Total Traffic (Bytes)' in result.columns)

    def test_top_customers_by_engagement(self):
        result = self.user_engagement.top_customers_by_engagement(n=3)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 3)  # 3 metrics
        for metric in result.values():
            self.assertEqual(len(metric), 3)  # top 3 customers

    def test_classify_customers_by_engagement(self):
        result = self.user_engagement.classify_customers_by_engagement()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)  # 5 unique users
        self.assertTrue('Cluster' in result.columns)

    def test_analyze_clusters(self):
        clusters = self.user_engagement.classify_customers_by_engagement()
        result = self.user_engagement.analyze_clusters(clusters)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 3)  # 3 clusters
        for cluster_stats in result.values():
            self.assertTrue('Sessions Frequency' in cluster_stats)
            self.assertTrue('Session Duration' in cluster_stats)
            self.assertTrue('Total Traffic (Bytes)' in cluster_stats)

    def test_top_users_per_application(self):
        result = self.user_engagement.top_users_per_application()
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 7)  # 7 applications
        for app_users in result.values():
            self.assertEqual(len(app_users), 5)  # top 10 users, but we only have 5 unique users in our test data

if __name__ == '__main__':
    unittest.main()