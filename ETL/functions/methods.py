import pandas as pd

# Map Yes/No to 1/0 for telco.csv
def map_yes_no_telco(df):
	yes_no_columns = [
		'senior_citizen',
		'dependents',
		'multiple_lines',
		'internet_service',
		'online_security',
		'online_backup',
		'device_protection_plan',
		'premium_tech_support',
		'streaming_tv',
		'streaming_movies'
	]
	for col in yes_no_columns:
		if col in df.columns:
			df[col] = df[col].map({'Yes': 1, 'No': 0})

			
	return df


# Map Yes/No to 1/0 for train.csv
def map_yes_no_train(df):
	yes_no_columns = [
		'senior_citizen',
		'dependents',
		'multiple_lines',
		'internet_service',
		'online_security',
		'online_backup',
		'device_protection_plan',
		'premium_tech_support',
		'streaming_tv',
		'streaming_movies'
	]
	for col in yes_no_columns:
		if col in df.columns:
			df[col] = df[col].astype(int)


	return df


# Cleaning Function
def clean_data(df):

	# handle missing values
	df = df.fillna(method='ffill')
	# remove duplicates
	df = df.drop_duplicates()
	return df


# Transform telco.csv
def transform_telco_data(df):

	# Map Yes/No to 1/0
	df = map_yes_no_telco(df)

	# Ensure numerical columns are properly typed
	numerical_columns = ['tenure', 'monthly_charges', 'total_charges']
	for col in numerical_columns:
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors='coerce')

	# Ordinal encode contract column
	contract_mapping = {'Month-to-Month': 0, 'One Year': 1, 'Two Year': 2}
	df['contract'] = df['contract'].map(contract_mapping)

	# Label encode payment method
	payment_mapping = {'Bank Withdrawal': 0, 'Credit Card': 1}
	df['payment_method'] = df['payment_method'].map(payment_mapping)

	# Binary encode churn_label to churn column
	if 'churn_label' in df.columns:

		df['churn'] = df['churn_label'].apply(lambda x: 1 if x == 'Yes' else 0)

		# Drop churn_label column
		return df.drop(columns=['churn_label'])


# transform train.csv
def transform_train_data(df):

	# map yes/no to 1/0
	df = map_yes_no_train(df)

	# Ordinal encode contract
	contract_mapping = {'Month-to-Month': 0, 'One Year': 1, 'Two Year': 2}
	df['contract'] = df['contract'].map(contract_mapping)

	# Label encode payment method
	payment_mapping = {'Bank Withdrawal': 0, 'Credit Card': 1}
	df['payment_method'] = df['payment_method'].map(payment_mapping)

	# Ensure numerical columns are properly typed
	numerical_columns = ['tenure', 'monthly_charges', 'total_charges']
	for col in numerical_columns:
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors='coerce')


	return df
