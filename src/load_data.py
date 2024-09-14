"""
load_data.py

This module provides a class to load data from a PostgreSQL database, clean it by handling missing values and duplicates,
and return the cleaned DataFrame.
"""

from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
 
class DataLoader:
    """
    A class to load and clean data from a PostgreSQL database.
    """

    def __init__(self):
        """
        Initializes the DataLoader with database connection parameters from environment variables.
        """
        load_dotenv()
        self.db_host = os.getenv('DB_HOST')
        self.db_name = os.getenv('DB_NAME')
        self.db_user = os.getenv('DB_USER')
        self.db_password = os.getenv('DB_PASSWORD')
        self.engine = create_engine(f'postgresql://{self.db_user}:{self.db_password}@{self.db_host}/{self.db_name}')

    def load_data(self, query: str) -> pd.DataFrame:
        """
        Loads data from PostgreSQL database using the provided SQL query.

        :param query: SQL query to fetch data.
        :return: DataFrame containing the loaded data.
        """
        return pd.read_sql(query, self.engine)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the DataFrame by removing duplicates, handling missing values,
        and performing mean imputation on numerical columns.

        :param df: DataFrame to be cleaned.
        :return: Cleaned DataFrame.
        """

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Placeholder for categorical data
        categorical_columns = [
            'IMSI', 'MSISDN/Number', 
            'Handset Type', 'Handset Manufacturer'
        ]
        
        df.dropna(subset=categorical_columns, inplace=True)

        # # Remove rows where 'Handset Type' or 'Handset Manufacturer' is 'undefined'
        # df = df[~df['Handset Type'].str.lower().eq('undefined')]
        # df = df[~df['Handset Manufacturer'].str.lower().eq('undefined')]

        df['Handset Type'] = df['Handset Type'].str.replace('undefined', 'Unknown', case=False)
        df['Handset Manufacturer'] = df['Handset Manufacturer'].str.replace('undefined', 'Unknown', case=False)

        df.rename(columns={'Dur. (ms)': 'Dur.(s)'}, inplace=True)

        # Mean imputation for numerical data
        mean_imputation_columns = [
            "Start ms", "End ms", "Dur.(s)", 
            "Avg RTT DL (ms)", "Avg RTT UL (ms)", "Avg Bearer TP DL (kbps)",
            "Avg Bearer TP UL (kbps)", "TCP DL Retrans. Vol (Bytes)",
            "TCP UL Retrans. Vol (Bytes)", "DL TP < 50 Kbps (%)",
            "50 Kbps < DL TP < 250 Kbps (%)", "250 Kbps < DL TP < 1 Mbps (%)",
            "DL TP > 1 Mbps (%)", "UL TP < 10 Kbps (%)",
            "10 Kbps < UL TP < 50 Kbps (%)", "50 Kbps < UL TP < 300 Kbps (%)",
            "UL TP > 300 Kbps (%)", "HTTP DL (Bytes)", "HTTP UL (Bytes)",
            "Activity Duration DL (ms)", "Activity Duration UL (ms)", "Dur. (ms).1",
            "Nb of sec with 125000B < Vol DL", "Nb of sec with 1250B < Vol UL < 6250B",
            "Nb of sec with 31250B < Vol DL < 125000B", "Nb of sec with 37500B < Vol UL",
            "Nb of sec with 6250B < Vol DL < 31250B", "Nb of sec with 6250B < Vol UL < 37500B",
            "Nb of sec with Vol DL < 6250B", "Nb of sec with Vol UL < 1250B",
            "Social Media DL (Bytes)", "Social Media UL (Bytes)",
            "Google DL (Bytes)", "Google UL (Bytes)", "Email DL (Bytes)",
            "Email UL (Bytes)", "Youtube DL (Bytes)", "Youtube UL (Bytes)",
            "Netflix DL (Bytes)", "Netflix UL (Bytes)", "Gaming DL (Bytes)",
            "Gaming UL (Bytes)", "Other DL (Bytes)", "Other UL (Bytes)",
            "Total UL (Bytes)", "Total DL (Bytes)"
        ]

        for column in mean_imputation_columns:
            if df[column].dtype in [np.float64, np.int64]:  # Ensure the column is numerical
                df[column] = df[column].fillna(df[column].mean())
        # Handle outliers
        def handle_outliers(df: pd.DataFrame, columns: list) -> pd.DataFrame:
            """
            Handle outliers in specified columns using Z-score method.
            
            :param df: DataFrame with numerical columns.
            :param columns: List of columns to check for outliers.
            :return: DataFrame with outliers handled.
            """
            from scipy import stats

            for column in columns:
                if column in df.columns:
                    # Calculate Z-scores
                    df['z_score'] = np.abs(stats.zscore(df[column].dropna()))
                    # Remove outliers where Z-score > 3
                    df = df[df['z_score'] <= 3]
                    df = df.drop(columns=['z_score'])
                    
            return df

        # Define columns where outlier removal is necessary
        numerical_columns = mean_imputation_columns
        # df = handle_outliers(df, numerical_columns)
        
        return df
