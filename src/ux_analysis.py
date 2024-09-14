import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class UserExperience:
    """
    A class to analyze and visualize user experience metrics in telecom data.
    """
    
    def __init__(self, df):
        """
        Initialize the UserExperience class with a DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing telecom data.
        """
        self.df = df

    def aggregate_metrics(self):
        """
        Aggregate experience metrics per customer (MSISDN/Number).

        Returns:
        pd.DataFrame: Aggregated metrics per customer.
        """
        agg_metrics = self.df.groupby('MSISDN/Number').agg({
            'TCP DL Retrans. Vol (Bytes)': 'sum',
            'TCP UL Retrans. Vol (Bytes)': 'sum',
            'Avg RTT DL (ms)': 'mean',
            'Avg RTT UL (ms)': 'mean',
            'Avg Bearer TP DL (kbps)': 'mean',
            'Avg Bearer TP UL (kbps)': 'mean',
            'Handset Type': 'first'  # Assuming handset type doesn't change per user
        }).reset_index()
        
        return agg_metrics

    def analyze_metric_extremes(self, metric, n=10):
        """
        Compute and list top, bottom, and most frequent values for a given metric.

        Parameters:
        metric (str): The name of the metric to analyze.
        n (int): The number of values to return for each category.

        Returns:
        dict: A dictionary containing top, bottom, and most frequent values.
        """
        sorted_values = self.df[metric].sort_values(ascending=False)
        value_counts = self.df[metric].value_counts()

        return {
            'top': sorted_values.head(n).tolist(),
            'bottom': sorted_values.tail(n).tolist(),
            'most_frequent': value_counts.head(n).index.tolist()
        }

    def plot_metric_extremes(self, metric, n=10):
        """
        Plot top, bottom, and most frequent values for a given metric.

        Parameters:
        metric (str): The name of the metric to plot.
        n (int): The number of values to plot for each category.
        """
        extremes = self.analyze_metric_extremes(metric, n)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

        ax1.bar(range(n), extremes['top'])
        ax1.set_title(f'Top {n} {metric} Values')
        ax1.set_xlabel('Rank')
        ax1.set_ylabel(metric)

        ax2.bar(range(n), extremes['bottom'])
        ax2.set_title(f'Bottom {n} {metric} Values')
        ax2.set_xlabel('Rank')
        ax2.set_ylabel(metric)

        ax3.bar(range(n), self.df[metric].value_counts().head(n).values)
        ax3.set_title(f'Most Frequent {n} {metric} Values')
        ax3.set_xlabel('Value')
        ax3.set_ylabel('Frequency')
        ax3.set_xticklabels(extremes['most_frequent'], rotation=45)

        plt.tight_layout()
        plt.show()

    def get_top_handsets(self, n=10):
        """
        Get the top n most common handset types.

        Parameters:
        n (int): The number of top handsets to return. Defaults to 10.

        Returns:
        list: The top n most common handset types.
        """
        return self.df['Handset Type'].value_counts().nlargest(n).index.tolist()

    def plot_throughput_distribution(self, top_n=10):
        """
        Plot the distribution of average throughput for the top n handset types.

        Parameters:
        top_n (int): The number of top handsets to plot. Defaults to 10.
        """
        agg_data = self.aggregate_metrics()
        top_handsets = self.get_top_handsets(top_n)
        
        # Filter data for top handsets
        filtered_data = agg_data[agg_data['Handset Type'].isin(top_handsets)]
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Handset Type', y='Avg Bearer TP DL (kbps)', data=filtered_data, order=top_handsets)
        plt.title(f'Distribution of Average Downlink Throughput for Top {top_n} Handset Types')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Handset Type', y='Avg Bearer TP UL (kbps)', data=filtered_data, order=top_handsets)
        plt.title(f'Distribution of Average Uplink Throughput for Top {top_n} Handset Types')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def plot_tcp_retransmission(self, top_n=10):
        """
        Plot the average TCP retransmission view for the top n handset types.

        Parameters:
        top_n (int): The number of top handsets to plot. Defaults to 10.
        """
        agg_data = self.aggregate_metrics()
        top_handsets = self.get_top_handsets(top_n)
        
        # Filter data for top handsets
        filtered_data = agg_data[agg_data['Handset Type'].isin(top_handsets)]
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Handset Type', y='TCP DL Retrans. Vol (Bytes)', data=filtered_data, order=top_handsets, estimator=np.mean)
        plt.title(f'Average Downlink TCP Retransmission Volume for Top {top_n} Handset Types')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.barplot(x='Handset Type', y='TCP UL Retrans. Vol (Bytes)', data=filtered_data, order=top_handsets, estimator=np.mean)
        plt.title(f'Average Uplink TCP Retransmission Volume for Top {top_n} Handset Types')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def perform_kmeans_clustering(self):
        """
        Perform k-means clustering (k=3) to segment users based on experience metrics.

        Returns:
        pd.DataFrame: DataFrame with original data and assigned clusters.
        """
        agg_data = self.aggregate_metrics()
        
        # Select features for clustering
        features = [
            'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)',
            'Avg RTT DL (ms)', 'Avg RTT UL (ms)',
            'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)'
        ]
        
        # Normalize the features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(agg_data[features])
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        agg_data['Cluster'] = kmeans.fit_predict(scaled_features)
        
        return agg_data

    def plot_kmeans_results(self):
        """
        Plot the results of k-means clustering.
        """
        clustered_data = self.perform_kmeans_clustering()
        
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x='Avg Bearer TP DL (kbps)',
            y='Avg Bearer TP UL (kbps)',
            hue='Cluster',
            style='Cluster',
            data=clustered_data
        )
        plt.title('K-means Clustering of Users based on Experience Metrics')
        plt.xlabel('Average Downlink Throughput (kbps)')
        plt.ylabel('Average Uplink Throughput (kbps)')
        plt.legend(title='Cluster')
        plt.tight_layout()
        plt.show()

        # Additional visualization: Parallel Coordinates Plot
        plt.figure(figsize=(15, 8))
        features = [
            'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)',
            'Avg RTT DL (ms)', 'Avg RTT UL (ms)',
            'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)'
        ]
        pd.plotting.parallel_coordinates(
            clustered_data[features + ['Cluster']],
            'Cluster',
            colormap=plt.cm.Set2
        )
        plt.title('Parallel Coordinates Plot of Clusters')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()