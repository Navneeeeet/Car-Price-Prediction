import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
#loading file csv file into panda
car_dataset = pd.read_csv('/content/car data.csv')
#inspecting first 5 rows of dataframe
car_dataset.head()
#checking data sets
car_dataset.shape
car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)
car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)
car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)
#geteing some info about dataset
car_dataset.info()
x=car_dataset.drop(['Car_Name','Selling_Price'],axis=1)
y=car_dataset['Selling_Price']
print(y)
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.1,random_state=2)
from sklearn.linear_model import LinearRegression
lin_reg_model = LinearRegression()
lin_reg_model.fit(x_train, y_train)
#prediction on training data
training_data_prediction = lin_reg_model.predict(x_train)
#R squarred error
error_score=metrics.r2_score(y_train,training_data_prediction)
print("R squared error : ",error_score)
plt.scatter(y_train,training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price vs Predicted Price")
plt.show()
test_data_prediction = lin_reg_model.predict(x_test)
error_score_test=metrics.r2_score(y_test,test_data_prediction)
print("R squared error : ",error_score_test)
plt.scatter(y_test,test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price vs Predicted Price")
plt.show()
lasso_reg_model = Lasso()
lasso_reg_model.fit(x_train, y_train)
training_data_prediction = lasso_reg_model.predict(x_train)
error_score=metrics.r2_score(y_train,training_data_prediction)
print("R squared error : ",error_score)
plt.scatter(y_train,training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price vs Predicted Price")
plt.show()
test_data_prediction = lasso_reg_model.predict(x_test)
error_score_test=metrics.r2_score(y_test,test_data_prediction)
print("R squared error : ",error_score_test)
plt.scatter(y_test,test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price vs Predicted Price")
plt.show()
