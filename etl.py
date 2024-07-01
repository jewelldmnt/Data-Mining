import pandas as pd

#extract 

file_paths = ['D:\\Github Repos\\Data-Mining\\telco.csv',
              'D:\\Github Repos\\Data-Mining\\churn_data.csv', 
              'D:\\Github Repos\\Data-Mining\\customer_churn_data.csv',
              'D:\\Github Repos\\Data-Mining\\train.csv']

for path in file_paths: 

    try: 
        df = pd.read_csv(path)
        print(f"Data from {path}: ")
        print(df.head())
        print("\n")

    except Exception as e: 
        print(f"Failed to load {path}: {e}")
