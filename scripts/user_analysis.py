import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the src directory to the Python path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, src_dir)

# Import required modules
from load_data import DataLoader
from user_eda import UserOverview
from user_engagement_analysis import UserEngagement

def main():
    # Load and clean data
    loader = DataLoader()
    df = loader.load_data("SELECT * FROM public.xdr_data")
    cleaned_df = loader.clean_data(df)

    # Create UserOverview and UserEngagement objects
    user_eda = UserOverview(cleaned_df)
    user_engagement = UserEngagement(cleaned_df)

    # User Overview Analysis
    print("Executing User Overview Analysis...")
    
    # Describe dataset
    print("\nDataset Description:")
    print(user_eda.describe_dataset())

    # Plot top handsets and manufacturers
    user_eda.plot_top_handset_types()
    user_eda.plot_top_handset_manufacturers()
    user_eda.plot_top_handsets_per_manufacturer()

    # Decile analysis
    print("\nTop 5 Deciles by Total Data:")
    print(user_eda.segment_and_compute_decile())

    # Univariate analysis
    print("\nUnivariate Analysis:")
    print(user_eda.univariate_analysis())
    user_eda.graphical_univariate_analysis()

    # Bivariate analysis
    print("\nBivariate Analysis:")
    bivariate_results = user_eda.bivariate_analysis()
    print("Correlation Matrix:")
    print(bivariate_results['correlation_matrix'])

    # Plot correlation matrix
    user_eda.plot_correlation_matrix()

    # PCA analysis
    print("\nPCA Analysis:")
    pca_results = user_eda.pca_analysis()
    print("Explained Variance Ratio:")
    print(pca_results['explained_variance'])
    print("\nPCA Loadings:")
    print(pca_results['loadings'])

    # User Engagement Analysis
    print("\nExecuting User Engagement Analysis...")

    # Top customers by engagement
    top_customers = user_engagement.top_customers_by_engagement(n=10)
    for metric, customers in top_customers.items():
        print(f"\nTop 10 customers by {metric}:")
        print(customers)

    # Classify customers by engagement
    customer_clusters = user_engagement.classify_customers_by_engagement()
    print("\nCustomer Clusters:")
    print(customer_clusters.head())

    # Analyze clusters
    cluster_stats = user_engagement.analyze_clusters(customer_clusters)
    print("\nCluster Statistics:")
    print(cluster_stats)

    # Top users per application
    top_users_per_app = user_engagement.top_users_per_application()
    for app, users in top_users_per_app.items():
        print(f"\nTop 10 users for {app}:")
        print(users)

    # Plot top applications
    user_engagement.plot_top_applications()

    # K-means optimal
    print("\nK-means Optimal Analysis:")
    user_engagement.k_means_optimal()

if __name__ == "__main__":
    main()