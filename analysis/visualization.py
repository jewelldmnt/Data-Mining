import matplotlib.pyplot as plt 
import seaborn as sns 
# visualization 

def create_visualization(df,columns): 
	count = 1

	df_filtered = df[columns]

	numerical_features = ['tenure_in_months', 'monthly_charge', 'total_charges']
	
	for feature in numerical_features:

		plt.figure(figsize=(10,6)) # This is not necessary
		sns.histplot(df_filtered[feature], kde=True)
		plt.title(f'{feature.capitalize()} Distribution')
		plt.xlabel(feature.capitalize())
		plt.ylabel('Frequency')

		# save figure
		plt.savefig(f'graphs/{count}_{feature}_distribution.png')
		# plt.show()
		count += 1

		# clean figure
		plt.clf()


	categorical_features = [
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
		'contract',
		'payment_method'
	]

	for feature in categorical_features:
		
		# Bar Chart
		# plt.figure(figsize=(10,6))
		# sns.countplot(data=df_filtered, x=feature)
		# plt.title(f'{feature.capitalize()} Distribution')
		# plt.ylabel('Count')
		# plt.savefig(f'graphs/{count}_bar_{feature}_distribution.png')

		# Pie Chart
		plt.pie(df_filtered[feature].value_counts(), labels=df_filtered[feature].value_counts().index, autopct=autopct_format(df_filtered[feature].value_counts()))
		plt.suptitle(f'{feature.capitalize()} Distribution', x=0.5)
		plt.title(f'Total Count: {df_filtered[feature].count()}', x=0.5)
		plt.savefig(f'graphs/{count}_pie_{feature}_distribution.png')

		# If you want to show the plot
		# plt.show()
		
		count += 1

		# clean figure
		plt.clf()
		plt.close()

def autopct_format(values):
	def my_format(pct):
		total = sum(values)
		val = int(round(pct*total/100.0))
		return '{:.1f}%\n({v:d})'.format(pct, v=val)
	return my_format