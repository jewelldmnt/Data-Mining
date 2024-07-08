import pandas as pd 

#descriptive statistics 

def descriptive_statistics(df, columns): 

    df_filtered = df[columns]

    print("Summary Statistics: ")
    print(df_filtered.describe())

    print("\nChurn Distribution: ")
    print(df['churn'].value_counts(normalize=True))

    #calculate and print mean, median and standard deviation
    print("\nMean: ")
    print(df_filtered.mean(numeric_only=True))

    print("\nMedian: ")
    print(df_filtered.median(numeric_only=True))

    print("\nStandard Deviation: ")
    print(df_filtered.std(numeric_only=True))


  