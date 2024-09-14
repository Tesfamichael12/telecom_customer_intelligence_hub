import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class UserOverview:
    """
    A class to perform Exploratory Data Analysis (EDA) on telecom data.
    """
    
    def __init__(self, df):
        """
        Initialize the EDA class with a DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing telecom data.
        """
        self.df = df

    def describe_dataset(self):
        """
        Provides a detailed description of the dataset, including mean, median,
        standard deviation, and other statistics for numerical columns.

        Returns:
        pd.DataFrame: DataFrame with descriptive statistics for numerical columns.
        """
        
        return self.df.describe()
    
    def plot_top_handset_types(self):
        """
        Plots the top 10 handset types by count.
        """
        plt.figure(figsize=(12, 6))
        self.df.groupby('Handset Type').size().sort_values(ascending=False).head(10).plot.bar()
        
        plt.title('Top 10 Handset Types')
        plt.xlabel('Handset Type')
        plt.ylabel('Count')
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()

    def plot_top_handset_manufacturers(self):
        """
        Plots the top 3 handset manufacturers by count.
        """
        plt.figure(figsize=(12, 6))
        self.df.groupby('Handset Manufacturer').size().sort_values(ascending=False).head(3).plot.bar()
        
        plt.title('Top 3 Handset Manufacturers')
        plt.xlabel('Handset Manufacturer')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_top_handsets_per_manufacturer(self):
        """
        Plots the top three handsets per top three manufacturers using a grouped bar chart.

        :param df: DataFrame containing 'Handset Manufacturer' and 'Handset Type' columns.
        """
        # Find the top three manufacturers
        top_manufacturers = self.df['Handset Manufacturer'].value_counts().head(3).index

        # Filter the DataFrame for these top manufacturers
        top_df = self.df[self.df['Handset Manufacturer'].isin(top_manufacturers)]

        # Find the top three handsets for each top manufacturer
        top_handsets = (
            top_df.groupby(['Handset Manufacturer', 'Handset Type'])
                .size()
                .reset_index(name='Count')
                .sort_values(['Handset Manufacturer', 'Count'], ascending=[True, False])
        )
        
        # Get the top three handsets per manufacturer
        top_handsets = top_handsets.groupby('Handset Manufacturer').head(5)

        # Plotting
        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=top_handsets,
            x='Handset Type',
            y='Count',
            hue='Handset Manufacturer',
            palette='viridis'
        )
        
        plt.title('Top 5 Handsets per Top 3 Manufacturers')
        plt.xlabel('Handset Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='Handset Manufacturer')
        plt.tight_layout()
        plt.show()

    def segment_and_compute_decile(self):
        """
        Segments the users into top five decile classes based on the total duration
        for all sessions and computes the total data (DL+UL) per decile class.

        Returns:
        pd.DataFrame: DataFrame with decile classes and total data (DL+UL) per decile.
        """
        # Ensure 'Dur.(s)' is numeric
        self.df['Dur.(s)'] = pd.to_numeric(self.df['Dur.(s)'], errors='coerce')
        
        # Calculate total duration per user
        user_durations = self.df.groupby('MSISDN/Number')['Dur.(s)'].sum().reset_index()
        
        # Segment users into decile classes
        user_durations['Decile'] = pd.qcut(user_durations['Dur.(s)'], 10, labels=False) + 1
        
        # Merge decile information back into the original DataFrame
        df_with_decile = pd.merge(self.df, user_durations[['MSISDN/Number', 'Decile']], on='MSISDN/Number')
        
        # Compute total DL and UL per decile class
        total_data_per_decile = df_with_decile.groupby('Decile').agg({
            'Total DL (Bytes)': 'sum',
            'Total UL (Bytes)': 'sum'
        }).reset_index()
        
        # Compute the total data (DL + UL) per decile class
        total_data_per_decile['Total Data (Bytes)'] = total_data_per_decile['Total DL (Bytes)'] + total_data_per_decile['Total UL (Bytes)']
        
        # Sort by total data in descending order and select the top 5 deciles
        top_5_deciles = total_data_per_decile.sort_values(by='Total Data (Bytes)', ascending=False).head(5)
        
        return top_5_deciles
    
    def univariate_analysis(self):
        """
        Conducts a non-graphical univariate analysis by computing dispersion 
        parameters for each quantitative variable in the dataset.

        Returns:
        pd.DataFrame: DataFrame with dispersion parameters (mean, median, 
                        variance, standard deviation, and range) for each 
                        quantitative variable.
        """
        # Select only quantitative variables except ignored (numeric types)
        colums_ignored = ['Bearer Id', 'IMSI', 'MSISDN/Number', 'IMEI']
        columns_picked = ['Dur.(s)', 'Total DL (Bytes)', 'Total UL (Bytes)']
        # quantitative_df = self.df.select_dtypes(include=[np.number])
        # quantitative_df.drop(colums_ignored, axis=1, inplace=True)
        quantitative_df = self.df[columns_picked]
        # Compute dispersion parameters
        analysis_df = pd.DataFrame({
            'Mean': quantitative_df.mean(),
            'Median': quantitative_df.median(),
            'Variance': quantitative_df.var(),
            'Standard Deviation': quantitative_df.std(),
            'Range': quantitative_df.max() - quantitative_df.min()
        })
        
        return analysis_df
    
    def graphical_univariate_analysis(self):
        """
        Conducts a graphical univariate analysis by generating suitable plots 
        for each quantitative variable in the dataset. The method creates 
        histograms, boxplots, and/or KDE plots based on the variable type.

        Returns:
        None
        """
       # Select only quantitative variables except ignored (numeric types)
        # colums_ignored = ["Dur.(s)"]
        # quantitative_df = self.df.select_dtypes(include=[np.number])
        # quantitative_df.drop(colums_ignored, axis=1, inplace=True)
        columns_picked = ['Dur.(s)', 'Total DL (Bytes)', 'Total UL (Bytes)']
        quantitative_df = self.df[columns_picked]
        
        try:
            plt.style.use('default')
        except FileNotFoundError:
            plt.style.use('seaborn')
        
        for column in quantitative_df.columns:
            plt.figure(figsize=(14, 6))
            
            # Histogram
            plt.subplot(1, 3, 1)
            sns.histplot(quantitative_df[column], kde=True)
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')

            # Boxplot
            plt.subplot(1, 3, 2)
            sns.boxplot(y=quantitative_df[column])
            plt.title(f'Boxplot of {column}')
            plt.ylabel(column)

            # KDE Plot
            plt.subplot(1, 3, 3)
            sns.kdeplot(quantitative_df[column], shade=True)
            plt.title(f'KDE Plot of {column}')
            plt.xlabel(column)
            plt.ylabel('Density')

            plt.tight_layout()
            plt.show()

        return quantitative_df

    def bivariate_analysis(self):
        """
        Perform bivariate analysis to explore the relationship between each application
        and the total download (DL) and upload (UL) data.

        Returns:
            dict: A dictionary containing correlation matrices and plots.
        """
        # Columns related to applications and total DL/UL
        application_cols = [
            "Social Media DL (Bytes)", "Social Media UL (Bytes)",
            "Google DL (Bytes)", "Google UL (Bytes)",
            "Email DL (Bytes)", "Email UL (Bytes)",
            "Youtube DL (Bytes)", "Youtube UL (Bytes)",
            "Netflix DL (Bytes)", "Netflix UL (Bytes)",
            "Gaming DL (Bytes)", "Gaming UL (Bytes)",
            "Other DL (Bytes)", "Other UL (Bytes)"
        ]
        total_dl_ul_cols = ["Total DL (Bytes)", "Total UL (Bytes)"]

        # Create a DataFrame for applications and total DL/UL
        app_total_df = self.df[application_cols + total_dl_ul_cols]

        # Compute correlation matrix
        correlation_matrix = app_total_df.corr()

        # Create plots for correlation matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
        plt.title('Correlation Matrix of Applications and Total Data (DL+UL)')
        plt.show()

        # Return results
        return {
            'correlation_matrix': correlation_matrix
        }
    def plot_correlation_matrix(self):
        """
        Computes the correlation matrix for application-related columns and plots it.

        This method selects columns related to application data, computes the correlation matrix,
        and visualizes it using a heatmap.
        """
        # List of application columns
        applications = [
            "Social Media DL (Bytes)", "Social Media UL (Bytes)",
            "Google DL (Bytes)", "Google UL (Bytes)",
            "Email DL (Bytes)", "Email UL (Bytes)",
            "Youtube DL (Bytes)", "Youtube UL (Bytes)",
            "Netflix DL (Bytes)", "Netflix UL (Bytes)",
            "Gaming DL (Bytes)", "Gaming UL (Bytes)",
            "Other DL (Bytes)"
        ]
        
        # Select the relevant columns
        app_data = self.df[applications]
        
        # Compute correlation matrix
        corr_matrix = app_data.corr()
        
        # Plot correlation matrix
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix of Application Data')
        plt.show()

        return corr_matrix

    def pca_analysis(self, n_components=2):
        """
        Perform Principal Component Analysis (PCA) to reduce the dimensions of the data
        and provide useful interpretation of the results.

        Args:
            n_components (int): The number of principal components to retain. Default is 2.

        Returns:
            dict: A dictionary containing PCA results, explained variance ratio, and loadings.
        """
        # Select numerical columns for PCA
        colums_ignored = ['Bearer Id', 'IMSI', 'MSISDN/Number', 'IMEI','Start ms','End ms',"Dur.(s)"]
        quantitative_df = self.df.select_dtypes(include=[np.number])
        quantitative_df.drop(colums_ignored, axis=1, inplace=True)
        numerical_cols = quantitative_df.columns.tolist()
        
           
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df[numerical_cols])
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_data)
        
        # Create a DataFrame for PCA results
        pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
        
        # Explained variance
        explained_variance = pca.explained_variance_ratio_
        
        # Create loadings DataFrame
        loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(n_components)],
                                index=numerical_cols)
        
        # Plot explained variance
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, n_components+1), explained_variance, 'o-', markersize=10)
        plt.title('Explained Variance Ratio by Principal Component')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.xticks(range(1, n_components+1))
        plt.grid(True)
        plt.show()
        
        # Plot the first two principal components if n_components >= 2
        if n_components >= 2:
            plt.figure(figsize=(10, 7))
            sns.scatterplot(x=pca_df['PC1'], y=pca_df['PC2'])
            plt.title('PCA: First vs Second Principal Component')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.grid(True)
            plt.show()

        # Return results
        return {
            'pca_df': pca_df,
            'explained_variance': explained_variance,
            'loadings': loadings
        }