## House Price Prediction

This repository contains code to predict house prices using a linear regression model. The dataset used for training and testing includes various features about the houses, and the model is trained to predict the sale prices based on these features.

# Installation

Before running the code, ensure you have the following packages installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Data

The dataset used in this project is split into training and test sets:
- `train.csv`: Contains the training data with features and the target variable `SalePrice`.
- `test.csv`: Contains the test data with features.

## Code Description

The main steps in the code are as follows:

1. **Importing Libraries**:
   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LinearRegression
   from sklearn.metrics import mean_squared_error
   ```

2. **Loading Data**:
   ```python
   train_data = pd.read_csv('train.csv')
   test_data = pd.read_csv('test.csv')
   ```

3. **Displaying Data**:
   Print the first few rows of the datasets:
   ```python
   print(train_data.head())
   print(test_data.head())
   ```

4. **Handling Missing Values**:
   Check for missing values in the data and handle them:
   ```python
   print(train_data.isnull().sum())
   print("Missing values in train data:")
   print(train_data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']].isnull().sum())

   print("Missing values in test data:")
   print(test_data[['GrLivArea', 'BedroomAbvGr', 'FullBath']].isnull().sum())

   train_data = train_data.dropna(subset=['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice'])
   test_data = test_data.dropna(subset=['GrLivArea', 'BedroomAbvGr', 'FullBath'])
   ```

5. **Preparing Data for Training**:
   Split the data into features and target variable:
   ```python
   X_train = train_data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
   y_train = train_data['SalePrice']
   X_test = test_data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
   ```

6. **Training the Model**:
   Train a linear regression model:
   ```python
   model = LinearRegression()
   model.fit(X_train, y_train)
   ```

7. **Making Predictions**:
   Predict house prices for the test set and calculate the Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) for the training set:
   ```python
   y_pred_test = model.predict(X_test)
   y_pred_train = model.predict(X_train)
   mse_train = mean_squared_error(y_train, y_pred_train)
   rmse_train = np.sqrt(mse_train)
   print(f"Mean Squared Error on Training Set: {mse_train}")
   print(f"Root Mean Squared Error on Training Set: {rmse_train}")
   ```

8. **Output Results**:
   Print the predictions, coefficients, and intercept of the model:
   ```python
   print("Predictions for Test Set:")
   print(y_pred_test)
   print("Coefficients:", model.coef_)
   print("Intercept:", model.intercept_)
   ```

## Results

- **Mean Squared Error on Training Set**: 2628535155.618378
- **Root Mean Squared Error on Training Set**: 51269.241808499355
- **Model Coefficients**: [110.06172639, -27859.33222353, 29694.68839062]
- **Model Intercept**: 47509.4821894654

## Conclusion

This code demonstrates a simple linear regression model to predict house prices based on a few selected features. Further improvements can be made by including more features, handling missing values more effectively, and experimenting with different models.
