from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def apply_ml_models(df):

	# Selecting specific features
	selected_features = [
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
		'payment_method',
		'tenure_in_months',
		'monthly_charge',
		'total_charges'
	]
	
	# Ensure selected features are present in df
	df_selected = df[selected_features + ['churn']]

	# Perform one-hot encoding for categorical variables
	df_encoded = pd.get_dummies(df_selected)

	x = df_encoded.drop(columns=['churn'])
	y = df_encoded['churn']
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

	rf_model = RandomForestClassifier(random_state=42)
	rf_model.fit(x_train, y_train)
	rf_preds = rf_model.predict(x_test)

	evaluate_model(y_test, rf_preds, model_name='Random Forest', classes=[0, 1])
	plot_feature_importance(rf_model, x_train.columns)

def evaluate_model(y_true, y_pred, model_name='Model', classes=None):

	accuracy = accuracy_score(y_true, y_pred)
	precision = precision_score(y_true, y_pred, average='binary', pos_label=1)
	recall = recall_score(y_true, y_pred, average='binary', pos_label=1)
	f1 = f1_score(y_true, y_pred, average='binary', pos_label=1)
	roc = roc_auc_score(y_true, y_pred)

	print(f"Metrics for {model_name}")
	print(f"Accuracy: {accuracy:.2f}")
	print(f"Precision: {precision:.2f}")
	print(f"Recall: {recall:.2f}")
	print(f"F1-score: {f1:.2f}")
	print(f"ROC AUC: {roc:.2f}")

	print(f"\nConfusion Matrix for {model_name}")

def plot_feature_importance(model, feature_names):
	feature_importances = pd.Series(model.feature_importances_, index=feature_names)
	feature_importances = feature_importances.sort_values(ascending=False)

	plt.figure(figsize=(12, 8))
	sns.barplot(x=feature_importances, y=feature_importances.index)
	plt.xlabel('Feature Importance Score')
	plt.ylabel('Features')
	plt.title('Random Forest Feature Importance')
	plt.show()
