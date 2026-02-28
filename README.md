# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1) Import pandas module and import the required data set.
2) Find the null values and count them.
3) Count number of left values.
4) From sklearn import LabelEncoder to convert string values to numerical values.
5) From sklearn.model_selection import train_test_split.
6) Assign the train dataset and test dataset.
7) From sklearn.tree import DecisionTreeClassifier.
8) Use criteria as entropy.
9) From sklearn import metrics.
10) Find the accuracy of our model and predict the require values.
```
## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Thaarakeshwar
RegisterNumber: 25014935 (212225040466)
```
```
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the data
df = pd.read_csv('Employee.csv')

# Display basic info
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Fix column name issue (there's a space in 'Departments ')
df.columns = df.columns.str.strip()  # Remove any whitespace from column names

# Encode categorical variables
le_department = LabelEncoder()
le_salary = LabelEncoder()

df['Department'] = le_department.fit_transform(df['Departments'])
df['Salary'] = le_salary.fit_transform(df['salary'])

# Drop the original categorical columns
df = df.drop(['Departments', 'salary'], axis=1)

# Separate features and target
X = df.drop('left', axis=1)  # Features (all except 'left')
y = df['left']                # Target (0 = stayed, 1 = left)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Make predictions
y_pred = dt_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Simple prediction for a sample employee
print("\n" + "="*50)
print("PREDICTION EXAMPLE")
print("="*50)

# Create a sample employee
sample_employee = pd.DataFrame([{
    'satisfaction_level': 0.4,
    'last_evaluation': 0.5,
    'number_project': 3,
    'average_montly_hours': 150,
    'time_spend_company': 3,
    'Work_accident': 0,
    'promotion_last_5years': 0,
    'Department': 5,  # Encoded value
    'Salary': 1        # Encoded value (0=low, 1=medium, 2=high)
}])

# Predict
prediction = dt_classifier.predict(sample_employee)[0]
probabilities = dt_classifier.predict_proba(sample_employee)[0]

print(f"Prediction: {'Will Leave' if prediction == 1 else 'Will Stay'}")
print(f"Probability of staying: {probabilities[0]:.2f}")
print(f"Probability of leaving: {probabilities[1]:.2f}")
```
## Output:
## DATASET SHAPE
<img width="781" height="481" alt="image" src="https://github.com/user-attachments/assets/3ec79a20-19b0-4ea0-8deb-ec2bc8f7fa4e" />

## MISSING VALUES
<img width="278" height="263" alt="image" src="https://github.com/user-attachments/assets/910b9cf2-0e5e-4696-9946-d3a9a0778c16" />

## ACCURACY
<img width="278" height="263" alt="image" src="https://github.com/user-attachments/assets/73747b18-a2c9-4196-aef9-6c7d6a775100" />

## CLASSIFICATION
<img width="534" height="205" alt="image" src="https://github.com/user-attachments/assets/204b490b-6ea5-4a8b-af91-1c490e813d82" />

## FEATURE IMPORTANCE
<img width="398" height="253" alt="image" src="https://github.com/user-attachments/assets/1d6f922c-3d72-407a-9be7-dc01dcc6cf6b" />

## FEATURE IMPORTANCE
<img width="519" height="128" alt="image" src="https://github.com/user-attachments/assets/57df289e-40e4-4fef-8198-e2441a3f1e57" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
