import pandas as pd
import sqlite3
from ETL.functions.methods import clean_data, transform_telco_data, transform_train_data


def etl(telco_path, train_path):
	# Extract
	telco_df, train_df = extract(telco_path, train_path)

	# Transform
	combined_df = transform(telco_df, train_df)

	# Load
	load(combined_df)


def extract(telco_path, train_path):
	print("Extract from CSV")

	# Load data
	telco_df = pd.read_csv(telco_path)
	train_df = pd.read_csv(train_path)

	print("Extract Successful")
	return telco_df, train_df


def transform(telco_df, train_df):
	print("Tranform to Combined Dataframe")

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

	print("Transform Successful")
	return combined_df


def load(combined_df):
	print("Load to SQLite Database")

	# Save to CSV (optional)
	# file_path = '..datasets/combined_data.csv'
	# combined_df.to_csv(file_path, index=False)
	
	# Initialize database connection
	try:
		conn = sqlite3.connect("combined_data.db")
	except sqlite3.Error as e:
		print(e)
	finally:
		# Clean table if exists	
		cursor = conn.cursor()
		cursor.execute("DROP TABLE IF EXISTS combined_data")
		cursor.close()

		# Save data to database
		combined_df.to_sql('combined_data', conn, if_exists='replace', index=False)

		if (conn):
			conn.close()


	print("Load Successful")


def get_data():
	# Load data from database
	try:
		conn = sqlite3.connect("combined_data.db")
	except sqlite3.Error as e:
		print(e)
	finally:
		query = "SELECT * FROM combined_data"
		data = pd.read_sql(query, conn)

		if (conn):
			conn.close()


	return data