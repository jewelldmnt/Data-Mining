from ETL.etl import etl, get_data
from analysis.descriptive_statistics import descriptive_statistics
from analysis.visualization import create_visualization
from analysis.ml_classifiers import apply_ml_models

# File paths for datasets
telco_path = 'datasets/telco.csv'
train_path = 'datasets/train.csv'

# Extract, Transform, and Load Data
etl(telco_path, train_path);

combined_df = get_data()

# Columns for descriptive statistics and visualization
columns_to_analyze = [
    'senior_citizen',
  		'dependents',
  		'multiple_lines',
  		'internet_service',
  		'online_security',
  		'online_backup',
  		'device_protection_plan',
  		'premium_tech_support',
  		'streaming_tv',
  		'streaming_movies',
  		'tenure_in_months',
  		'monthly_charge',
  		'total_charges',
  		'contract',
  		'payment_method'
]

# Perform EDA
descriptive_statistics(combined_df, columns_to_analyze)
create_visualization(combined_df, columns_to_analyze)

# Apply Machine Learning models and evaluate
# apply_ml_models(combined_df)