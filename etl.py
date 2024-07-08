import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

# file paths
telco_path = 'D:\\Github Repos\\Data-Mining\\telco.csv'
train_path = 'D:\\Github Repos\\Data-Mining\\train.csv'

# load data
telco_df = pd.read_csv(telco_path)
train_df = pd.read_csv(train_path)

# standardize column names 
telco_df.columns = telco_df.columns.str.strip().str.lower().str.replace(' ', '_')
train_df.columns = train_df.columns.str.strip().str.lower().str.replace(' ', '_')

# map yes/no to 1/0 for telco.csv
def map_yes_no_telco(df):
    yes_no_columns = [
        'senior_citizen', 'dependents', 'multiple_lines', 'internet_service',
        'online_security', 'online_backup', 'device_protection_plan', 'premium_tech_support',
        'streaming_tv', 'streaming_movies'
    ]
    for col in yes_no_columns:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    return df

# map yes/no to 1/0 for train.csv
def map_yes_no_train(df):
    yes_no_columns = [
        'senior_citizen', 'dependents', 'multiple_lines', 'internet_service',
        'online_security', 'online_backup', 'device_protection_plan', 'premium_tech_support',
        'streaming_tv', 'streaming_movies'
    ]
    for col in yes_no_columns:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df

# cleaning function

def clean_data(df):

    # handle missing values
    df = df.fillna(method='ffill')
    # remove duplicates
    df = df.drop_duplicates()
    return df

# transform telco.csv

def transform_telco_data(df):

    # map yes/no to 1/0
    df = map_yes_no_telco(df)
    
    # ensure numerical columns are properly typed
    numerical_columns = ['tenure', 'monthly_charges', 'total_charges']
    for col in numerical_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # ordinal encode contract column
    contract_mapping = {'Month-to-Month': 0, 'One Year': 1, 'Two Year': 2}
    df['contract'] = df['contract'].map(contract_mapping)
    
    # label encode payment method
    payment_mapping = {'Bank Withdrawal': 0, 'Credit Card': 1}
    df['payment_method'] = df['payment_method'].map(payment_mapping)
    
    # Binary encode churn_label to churn column
    df['churn'] = df['churn_label'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    return df.drop(columns=['churn_label'])  # Drop churn_label column

# transform train.csv 

def transform_train_data(df):

    # map yes/no to 1/0
    df = map_yes_no_train(df)
    
    # Ordinal encode contract
    contract_mapping = {'Month-to-Month': 0, 'One Year': 1, 'Two Year': 2}
    df['contract'] = df['contract'].map(contract_mapping)
    
    # label encode payment method
    payment_mapping = {'Bank Withdrawal': 0, 'Credit Card': 1}
    df['payment_method'] = df['payment_method'].map(payment_mapping)
    
    # ensure numerical columns are properly typed
    numerical_columns = ['tenure', 'monthly_charges', 'total_charges']
    for col in numerical_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# clean and transform data 
telco_df = transform_telco_data(clean_data(telco_df))
train_df = transform_train_data(clean_data(train_df))

# ensure same columns for both df after transformation 

telco_df = telco_df.reindex(columns=train_df.columns, fill_value=0)


# merge datasets
combined_df = pd.concat([telco_df, train_df], ignore_index=True)

# save to csv
combined_df.to_csv('D:\\Github Repos\\Data-Mining\\combined_data.csv', index=False)

print("ETL pipeline completed successfully.")

#columns for descriptive statistics and visualization 

columns_to_analyze = [
    'senior_citizen', 'dependents', 'multiple_lines', 'internet_service',
    'online_security', 'online_backup', 'device_protection_plan', 'premium_tech_support',
    'streaming_tv', 'streaming_movies', 'tenure', 'monthly_charges', 'total_charges',
    'contract', 'payment_method'
]

#descriptive statistics 

def descriptive_statistics(df): 

    print("Summary Statistics: ")
    print(df.describe())

    print("\nChurn Distribution: ")
    print(df['churn'].value_counts(normalize=True))

    #calculate and print mean, median and standard deviation
    print("\nMean: ")
    print(df.mean(numeric_only=True))

    print("\nMedian: ")
    print(df.median(numeric_only=True))

    print("\nStandard Deviation: ")
    print(df.std(numeric_only=True))

# visualization 

def plot_histograms(df): 

    df.hist(bins=30, figsize=(20,15))
    plt.show()

def plot_boxplots(df):

    numeric_features = df.select_dtypes(include=[np.number])

    for column in numeric_features.columns:
        plt.figure(figsize=(10,5))
        sns.boxplot(x='churn', y=column, data=df)
        plt.title(f'Box plot of {column} by churn')
        plt.show()

def plot_bar_charts(df):

    categorical_features = df.select_dtypes(include=[object])

    for column in categorical_features:
        plt.figure(figsize=(10,5))
        sns.countplot(x=column, hue='churn', data=df)
        plt.title(f"Bar chart of {column} by churn")
        plt.show()

def correlation_analysis(df):

    correlation_matrix = df.corr()
    plt.figure(figsize=(20,15))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.show()

    print("Correlation with churn: ")
    print(correlation_matrix['churn'].sort_values(ascending=False))

#perform EDA 

descriptive_statistics(combined_df)
plot_histograms(combined_df)
plot_boxplots(combined_df)
correlation_analysis(combined_df)
