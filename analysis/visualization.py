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
		plt.savefig(f'graphs/{count}_{feature}_distribution.png', bbox_inches='tight')
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

		# Clean Text: change underscores to spaces and capitalize first letter of each word
		cleaned_feature = feature.replace('_', ' ').title()

		# Add legend
		if feature in [
			'multiple_lines',
			'internet_service',
			'online_security',
			'online_backup',
			'device_protection_plan',
			'premium_tech_support',
		]:
			plt.legend([f'Did not avail {cleaned_feature}', f'{cleaned_feature}'],
				title="Legend", loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)


		if feature in ['dependents']:
			plt.legend([f'No {cleaned_feature}', f'{cleaned_feature}'],
				title="Legend", loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)


		if feature in ['senior_citizen', 'streaming_tv', 'streaming_movies']:
			plt.legend([f'Not {cleaned_feature}', f'{cleaned_feature}'],
				title="Legend", loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)


		if feature in ['contract']:
			plt.legend(['Month-to-Month', 'One Year', 'Two Year'],
				title="Legend", loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)

		
		if feature in ['payment_method']:
			plt.legend(['Bank Withdrawal', 'Credit Card', 'Mailed Check'], 
				title="Legend", loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)


		plt.suptitle(f'{cleaned_feature} Distribution', x=0.5)
		plt.title(f'Total Count: {df_filtered[feature].count()}', x=0.5)
		plt.savefig(
			f'graphs/{count}_pie_{feature}_distribution.png', bbox_inches='tight')

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