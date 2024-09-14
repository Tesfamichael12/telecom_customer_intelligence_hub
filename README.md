# Telecom Customer Intelligence Hub - Data Analysis and Machine Learning

This repository contains a comprehensive analysis of customer data for a telecommunications company using data science and machine learning techniques. The project aims to provide actionable insights into customer behavior, engagement, experience, and satisfaction to drive business growth and enhance customer services.

## Project Overview

The objective of this project is to perform an in-depth analysis of telecom customer data to help identify opportunities for improving user experience, engagement, and overall satisfaction. The project also involves building predictive models to gain insights into customer behavior and deploying an interactive dashboard for data visualization and reporting.

## Key Features

- **Data Extraction & Preprocessing:** Extraction of telecom customer data from PostgreSQL databases, followed by thorough cleaning, transformation, and formatting for analysis.
- **Exploratory Data Analysis (EDA):** Insights into customer behavior through univariate, bivariate, and multivariate analyses, including data visualizations using libraries like `matplotlib` and `seaborn`.
- **User Overview Analysis:** Identification of top handsets, manufacturers, and data usage patterns to understand customer preferences and trends.
- **User Engagement Analysis:** Calculation of engagement metrics (session frequency, duration, and traffic) to classify customers based on their usage patterns using clustering algorithms.
- **User Experience Analysis:** Examination of network parameters (TCP retransmission, RTT, throughput) and handset characteristics to assess customer experience.
- **Customer Satisfaction Analysis:** Development of an engagement and experience score to derive a customer satisfaction score using machine learning models.
- **Machine Learning Models:** Implementation of clustering (K-means) and regression models to predict customer satisfaction and segment users.
- **Dashboard Development:** Creation of an interactive and user-friendly dashboard using Streamlit to visualize data insights and provide a detailed analysis.
- **Model Deployment & Tracking:** Use of Docker and MLFlow for model deployment, tracking, and versioning to monitor changes in the models.

## Repository Structure

- **/src:** Contains the main Python scripts for data processing, analysis, modeling, and dashboard creation.
- **/notebooks:** Jupyter notebooks for exploratory data analysis, model development, and documentation of insights.
- **/scripts:** Standalone scripts for various tasks like data extraction, preprocessing, and model training.
- **/tests:** Unit tests to ensure code robustness and reliability.
- **/dashboard:** Streamlit dashboard files for visualizing the analysis results.
- **/data:** Placeholder for datasets used in the project.
- **/models:** Saved models for customer engagement, experience, and satisfaction analysis.
- **requirements.txt:** List of Python dependencies for the project.
- **README.md:** Project overview and usage instructions.
- **Dockerfile:** Docker configuration for containerizing the application.
- **.github/workflows:** CI/CD setup using GitHub Actions for automated testing and deployment.

## How to Use

1. **Installation:** Clone the repository and install the dependencies listed in `requirements.txt`.
2. **Data Extraction:** Use the provided scripts to extract data from the PostgreSQL database.
3. **Data Analysis:** Run the notebooks or scripts for exploratory data analysis and feature engineering.
4. **Modeling:** Train models for customer engagement and satisfaction analysis using the scripts in the `/src` directory.
5. **Dashboard:** Run the Streamlit dashboard to visualize insights.
6. **Deployment:** Use the Dockerfile to containerize the application and deploy it using Docker or another hosting service.

## Getting Started

- Clone the repository:
  ```bash
  git clone https://github.com/Tesfamichael12/telecom_customer_intelligence_hub.git
  ```
