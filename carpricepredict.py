import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import metrics

# Load dataset
file_path = '/content/car_data.csv'  # Ensure correct path
try:
    car_dataset = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")

# Inspect dataset
print(car_dataset.head())
print(car_dataset.shape)
print(car_dataset.info())

# Handle missing values
print("Missing values:\n", car_dataset.isnull().sum())

# Encode categorical variables
car_dataset.replace({'Seller_Type': {'Dealer': 0, 'Individual': 1}}, inplace=True)
car_dataset.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)
car_dataset.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)

# Ensure required columns exist
if 'Car_Name' in car_dataset.columns and 'Selling_Price' in car_dataset.columns:
    X = car_dataset.drop(['Car_Name', 'Selling_Price'], axis=1)
    y = car_dataset['Selling_Price']
else:
    print("Error: Required columns not found in dataset")
    exit()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

# Linear Regression
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, y_train)

train_pred_lr = lin_reg_model.predict(X_train)
test_pred_lr = lin_reg_model.predict(X_test)

train_error_lr = metrics.r2_score(y_train, train_pred_lr)
test_error_lr = metrics.r2_score(y_test, test_pred_lr)

print("Linear Regression R squared error (Train):", train_error_lr)
print("Linear Regression R squared error (Test):", test_error_lr)

plt.scatter(y_train, train_pred_lr, label='Train', color='blue', alpha=0.5)
plt.scatter(y_test, test_pred_lr, label='Test', color='red', alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price (Linear Regression)")
plt.legend()
plt.show()

# Lasso Regression
lasso_model = Lasso()
lasso_model.fit(X_train, y_train)

train_pred_lasso = lasso_model.predict(X_train)
test_pred_lasso = lasso_model.predict(X_test)

train_error_lasso = metrics.r2_score(y_train, train_pred_lasso)
test_error_lasso = metrics.r2_score(y_test, test_pred_lasso)

print("Lasso Regression R squared error (Train):", train_error_lasso)
print("Lasso Regression R squared error (Test):", test_error_lasso)

plt.scatter(y_train, train_pred_lasso, label='Train', color='blue', alpha=0.5)
plt.scatter(y_test, test_pred_lasso, label='Test', color='red', alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price (Lasso Regression)")
plt.legend()
plt.show()
