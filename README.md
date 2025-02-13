# Car-Price-Prediction
# Second-Hand Car Price Prediction

## Overview
This project aims to predict the selling price of second-hand cars using **Linear Regression** and **Lasso Regression** models. The dataset consists of various features such as car type, fuel type, seller type, transmission type, and more.

## Dataset
The dataset used is `car_data.csv`, which should be placed in the `data/` directory. The dataset includes features like:
- `Year`: Manufacturing year of the car.
- `Selling_Price`: The price at which the car is being sold.
- `Present_Price`: The current price of the car.
- `Kms_Driven`: Total kilometers driven by the car.
- `Fuel_Type`: Type of fuel used (Petrol, Diesel, CNG).
- `Seller_Type`: Whether the seller is a dealer or an individual.
- `Transmission`: Manual or automatic transmission.
- `Owner`: Number of previous owners.

## Installation
To run this project, install the required dependencies:
```sh
pip install -r requirements.txt
```

## Usage
Run the model script using:
```sh
python src/model.py
```

## Model Training & Evaluation
The project trains two regression models:
- **Linear Regression**
- **Lasso Regression**

### Performance Metrics:
- **Linear Regression R² Score (Train & Test)**
- **Lasso Regression R² Score (Train & Test)**

## Features & Improvements
- Handles missing values and categorical encoding.
- Performs train-test split for better evaluation.
- Visualizes the actual vs predicted price for both models.

## Visualizations
The project includes scatter plots comparing actual vs predicted prices for both models.

## Future Improvements
- Implement more advanced regression models.
- Tune hyperparameters for better accuracy.
