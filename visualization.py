import matplotlib.pyplot as plt 
import seaborn as sns 
# visualization 

def create_visualization(df,columns): 

    df_filtered = df[columns]

    numerical_features = ['tenure_in_months', 'monthly_charge', 'total_charges']
    
    for feature in numerical_features: 
        plt.figure(figsize=(10,6))
        sns.histplot(df_filtered[feature], kde=True)
        plt.title(f'{feature.capitalize()} Distribution')
        plt.xlabel(feature.capitalize())
        plt.ylabel('Frequency')
        plt.show()

    #bar plots for categorical features 

    categorical_features = [
        'senior_citizen', 'dependents', 'multiple_lines', 'internet_service',
        'online_security', 'online_backup', 'device_protection_plan', 'premium_tech_support',
        'streaming_tv', 'streaming_movies', 'contract', 'payment_method'
    ]

    for feature in categorical_features:
            plt.figure(figsize=(10,6))
            sns.countplot(data=df_filtered, x=feature)
            plt.title(f'{feature.capitalize()} Distribution')
            plt.ylabel('Count')
            plt.show()