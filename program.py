import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import io
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
df = iris.frame
print(os.getcwd())

# Load the dataset
from google.colab import files
uploaded = files.upload()

df = pd.read_csv(io.BytesIO(uploaded['beula(2).csv']))
print(uploaded.keys())

# Display the first few rows
print(df.head())

# Display column names
print(df.columns.tolist())
df.columns = df.columns.str.strip()

# Drop rows with missing target values
if 'Casualty_Severity' in df.columns:
    df = df.dropna(subset=['Casualty_Severity'])
    print("Rows with missing 'Casualty_Severity' dropped.")
else:
    print("'Casualty_Severity' column not found.")


# Fill missing values in other columns if necessary
df = df.fillna(method='ffill')

# Convert categorical variables to dummy/indicator variables
categorical_cols = ['Weather_Condition', 'Road_Type']
existing_cols = [col for col in categorical_cols if col in df.columns]

if existing_cols:
    df = pd.get_dummies(df, columns=categorical_cols)
else:
    print(f"No matching columns found among {categorical_cols}")


# Define features and target variable
if 'Casualty_Severity' in df.columns:
    X = df.drop('Casualty_Severity', axis=1)
    y = df['Casualty_Severity']
else:
    print("'Casualty_Severity' column is missing; skipping related operations.")


# Split into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)
# Predict on the test set
y_pred = rf_model.predict(X_test)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# Save the trained model to a file
joblib.dump(rf_model, 'random_forest_model.pkl')
# Load the model from the file
loaded_model = joblib.load('random_forest_model.pkl')

# Example: Predicting on new data
# new_data should be a DataFrame with the same structure as X
# new_data = pd.read_csv('new_data.csv')  # Replace with actual data
# predictions = loaded_model.predict(new_data)
# print(predictions)
