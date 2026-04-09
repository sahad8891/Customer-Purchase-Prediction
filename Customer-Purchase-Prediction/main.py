import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# -----------------------------------
# 1. SETUP AND LOAD DATA
# -----------------------------------

# Ensure screenshots directory exists
if not os.path.exists('screenshots'):
    os.makedirs('screenshots')

# Load the dataset
dataset_path = 'dataset.csv'
if not os.path.exists(dataset_path):
    print(f"Error: {dataset_path} not found. Please ensure the dataset exists.")
    exit()

df = pd.read_csv(dataset_path)

print("Dataset Preview:")
print(df.head())

# Features and Target
X = df.iloc[:, [1, 2]].values  # Age and EstimatedSalary
y = df.iloc[:, 3].values       # Purchased

# -----------------------------------
# 2. DATA PREPROCESSING
# -----------------------------------

# Train-Test Split (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Feature Scaling (Crucial for Logistic Regression)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# -----------------------------------
# 3. MODEL TRAINING
# -----------------------------------

classifier = LogisticRegression(random_state=42)
classifier.fit(X_train, y_train)

# -----------------------------------
# 4. PREDICTION AND EVALUATION
# -----------------------------------

y_pred = classifier.predict(X_test)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix and Classification Report
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------------
# 5. VISUALIZATION
# -----------------------------------

# Set aesthetic style
sns.set_theme(style="whitegrid")

# Graph 1: Buy vs Not Buy predictions (Bar Chart)
plt.figure(figsize=(8, 6))
unique, counts = np.unique(y_pred, return_counts=True)
sns.barplot(x=['Not Buy (0)', 'Buy (1)'], y=counts, hue=['Not Buy (0)', 'Buy (1)'], palette='viridis', legend=False)
plt.title('Prediction Results: Buy vs Not Buy')
plt.ylabel('Number of Customers')
plt.savefig('screenshots/prediction_chart.png')
print("Saved: screenshots/prediction_chart.png")
# plt.show() # Uncomment if running in interactive env

# Graph 2: Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Buy', 'Buy'], yticklabels=['Not Buy', 'Buy'])
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('screenshots/confusion_matrix.png')
print("Saved: screenshots/confusion_matrix.png")

# Graph 3: Accuracy Graph (Percentage)
plt.figure(figsize=(6, 4))
plt.bar(['Model Accuracy'], [accuracy * 100], color='teal')
plt.ylim(0, 100)
plt.ylabel('Percentage (%)')
plt.title('Model Performance (Accuracy)')
for i, v in enumerate([accuracy * 100]):
    plt.text(i, v + 2, f"{v:.2f}%", ha='center', fontweight='bold')
plt.savefig('screenshots/accuracy_graph.png')
print("Saved: screenshots/accuracy_graph.png")

# -----------------------------------
# 6. USER INPUT FEATURE
# -----------------------------------

print("\n" + "="*40)
print("   CUSTOMER PURCHASE PREDICTION")
print("="*40)

try:
    user_age = float(input("Enter Customer Age: "))
    user_salary = float(input("Enter Customer Estimated Salary: "))

    # Transform input using the same scaler
    user_input = sc.transform([[user_age, user_salary]])
    user_prediction = classifier.predict(user_input)

    if user_prediction[0] == 1:
        print("\nPrediction: The customer will likely BUY the product! (Yes)")
    else:
        print("\nPrediction: The customer is unlikely to buy the product. (No)")
except ValueError:
    print("Invalid input. Please enter numeric values for age and salary.")

print("="*40)
