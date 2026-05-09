import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path

# Step 1: Load the Dataset
script_dir = Path(__file__).resolve().parent
data_path = script_dir / 'shopping_data.csv'
data = pd.read_csv(data_path)

# Step 2: Exploratory Data Analysis (EDA)
print("First few rows of the dataset:")
print(data.head())

# Check for missing values
data = data.dropna()

# Convert categorical columns to numerical
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
data['DeviceType'] = data['DeviceType'].map({'Mobile': 0, 'Desktop': 1, 'Tablet': 2})
data['PurchaseIntent'] = data['PurchaseIntent'].map({'Yes': 1, 'No': 0})

# Separate features and target variable
X = data.drop(['UserID', 'PurchaseIntent'], axis=1)
y = data['PurchaseIntent']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy Score:", accuracy)

# Plots
# Plot 1: Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Plot 2: Distribution of Purchase Intent
plt.figure(figsize=(6, 4))
sns.countplot(data['PurchaseIntent'], palette='viridis')
plt.title('Distribution of Purchase Intent')
plt.xlabel('Purchase Intent')
plt.ylabel('Count')
plt.show()

# Plot 3: Feature Importance
feature_importances = model.feature_importances_
plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importances, y=data.columns[1:-1], palette='plasma')
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Plot 4: Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()