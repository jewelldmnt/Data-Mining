import pandas as pd
from etl import clean_data, transform_telco_data, transform_train_data
from descriptive_statistics import descriptive_statistics
from visualization import create_visualization
from ml_classifiers import apply_ml_models

# File paths
telco_path = 'D:\\Github Repos\\Data-Mining\\telco.csv'
train_path = 'D:\\Github Repos\\Data-Mining\\train.csv'

# Load data
telco_df = pd.read_csv(telco_path)
train_df = pd.read_csv(train_path)

# Standardize column names
telco_df.columns = telco_df.columns.str.strip().str.lower().str.replace(' ', '_')
train_df.columns = train_df.columns.str.strip().str.lower().str.replace(' ', '_')

# Clean and transform data
telco_df = transform_telco_data(clean_data(telco_df))
train_df = transform_train_data(clean_data(train_df))

# Ensure same columns for both dataframes after transformation
telco_df = telco_df.reindex(columns=train_df.columns, fill_value=0)

# Merge datasets
combined_df = pd.concat([telco_df, train_df], ignore_index=True)

# Save to CSV (optional)
file_path = 'D:\\Github Repos\\Data-Mining\\combined_data.csv'

combined_df.to_csv(file_path, index=False)

# Columns for descriptive statistics and visualization
columns_to_analyze = [
    'senior_citizen', 'dependents', 'multiple_lines', 'internet_service',
    'online_security', 'online_backup', 'device_protection_plan', 'premium_tech_support',
    'streaming_tv', 'streaming_movies', 'tenure_in_months', 'monthly_charge', 'total_charges',
    'contract', 'payment_method'
]

# Perform EDA
descriptive_statistics(combined_df, columns_to_analyze)
create_visualization(combined_df, columns_to_analyze)

# Apply Machine Learning models and evaluate
apply_ml_models(combined_df)
