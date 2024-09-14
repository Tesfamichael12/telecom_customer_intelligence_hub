# telecom_customer_intelligence_hub
A client is interested in purchasing TellCo, an existing mobile service provider in the Republic of Pefkakia. TellCo’s current owners have been willing to share their financial information but have never employed anyone to look at the data that is generated automatically by their systems.

The objective is to provide the client a report to analyse opportunities for growth and make a recommendation on whether TellCo is worth buying or selling.

Table of Contents
Installation
Project Structure 
Usage
Exploratory Data Analysis (EDA) Overview
Data Cleaning
Dashboard
Contributing
License
Installation
To set up this project on your local machine, follow the steps below:

Clone the repository:
git clone https://github.com/brukGit/tenx-w3.git
cd notebooks
Checkout branch task-1:
 git checkout task-1

3. **Create a virtual environment (optional but recommended)**:
  ```bash
  python3 -m venv venv
  source venv\Scripts\activate  # On Linux, use `venv/bin/activate`

4. **Install the required dependencies**:
  ```bash
  pip install -r requirements.txt


## Project Structure
  ```bash
      ├── data/                  # Directory containing raw datasets
      ├── notebooks/             # Jupyter notebooks for EDA and analysis
      ├── scripts/               # Python scripts for data processing and visualization
      ├── tests/                 # Unit tests for the project
      ├── app/                   # Interactive app built with streamlit
      ├── .github/workflows/     # GitHub Actions for CI/CD
      ├── .vscode/               # Visual Studio Code settings and configurations
      ├── requirements.txt       # Python dependencies
      ├── README.md              # Project documentation (this file)
      └── LICENSE                # License for the project


## Usage
### Running the Notebooks
To perform the EDA, navigate to the notebooks/ directory and open the provided Jupyter notebook. The notebook focuses analyzing both user overview and user engagement. 
  ```bash
  jupyter notebook notebooks/user_overview.ipynb
  jupyter notebook notebooks/user_engagement.ipynb
  jupyter notebook notebooks/user_experience.ipynb
  jupyter notebook notebooks/user_satisfaction.ipynb
 

### Running the scripts
You can use the script 'user_analysis.py' inside scripts directory to run all scripts located in 'src/' directory. Just change directory to scripts and executed the script inside. 
  ```bash
  cd scripts
  python user_analysis.py

### Running the app locally
Just change directory to 'app' and executed the script inside. 
  ```bash
  cd app
  python main.py


### Running Tests
If you want to run unit tests to ensure that the analysis classes and functions work as expected, run the following command in the root directory:
  
```bash
  python -m unittest discover -s tests

### Raw datasets
The raw datasets are fetched from a PostgreSQL.

## Exploratory Data Analysis (EDA) Overview
The EDA conducted in this project covers several key areas:

○	Univariate & Bivariate Analysis: Explored central tendencies, dispersion, and relationships between session metrics (duration, traffic) and applications (DL+UL data).

○	Outlier Detection: Identified and handled outliers in session durations and data volumes.

○	Customer Engagement Metrics: Aggregated session frequency, duration, and total traffic per customer for insights into user behavior.

○	Clustering: Used K-Means (with optimal clusters determined via the elbow method) to segment users by engagement levels.

○	Dimensionality Reduction: Applied PCA to simplify analysis while retaining data variance.

○	Application Engagement: Ranked top 10 users per application by data traffic.

○	Top Application Analysis: Visualized top 3 applications by usage through bar charts.

## Data Cleaning
Based on the initial analysis, the dataset was cleaned by handling missing values, removing duplicates, and ensuring correct data types.

## Dashboard
https://telco-tenx-w2.onrender.com/

## Contributing
Contributions to this project are welcome! If you have suggestions or improvements, feel free to open a pull request or issue on GitHub.

## License
This project is licensed under the MIT License - see the LICENSE file for details.