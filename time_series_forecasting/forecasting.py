import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA  # Updated import statement
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Generate sample time series data
np.random.seed(42)
n = 100
time = np.arange(n)
data = 10 + 0.5 * time + 5 * np.sin(time / 10) + np.random.normal(scale=2, size=n)

# Split data into train and test sets
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Define Evaluation Metrics
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# ARIMA Model
arima_model = ARIMA(train, order=(5, 1, 0))  # (p, d, q) parameters
arima_fit = arima_model.fit()
arima_predictions = arima_fit.forecast(steps=len(test))

# Calculate ARIMA Metrics
arima_rmse = calculate_rmse(test, arima_predictions)
arima_mae = calculate_mae(test, arima_predictions)
arima_mape = calculate_mape(test, arima_predictions)

# Prepare data for Linear Regression
X_train = time[:train_size].reshape(-1, 1)
y_train = train
X_test = time[train_size:].reshape(-1, 1)
y_test = test

# Train Linear Regression Model
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
linear_predictions = linear_regressor.predict(X_test)

# Calculate Linear Regression Metrics
linear_rmse = calculate_rmse(y_test, linear_predictions)
linear_mae = calculate_mae(y_test, linear_predictions)
linear_mape = calculate_mape(y_test, linear_predictions)

# Prepare data for Neural Network
X = np.arange(len(data)).reshape(-1, 1)
y = data

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Neural Network Model
model = Sequential([
    Dense(50, activation='relu', input_shape=(1,)),
    Dense(25, activation='relu'),
    Dense(1)
])

# Compile Model
model.compile(optimizer=Adam(), loss='mean_squared_error')

# Train Model
model.fit(X_train, y_train, epochs=50, verbose=0)

# Predict using Neural Network
nn_predictions = model.predict(X_test).flatten()

# Calculate Neural Network Metrics
nn_rmse = calculate_rmse(y_test, nn_predictions)
nn_mae = calculate_mae(y_test, nn_predictions)
nn_mape = calculate_mape(y_test, nn_predictions)

# Visualization
plt.figure(figsize=(12, 8))

# Plot Actual vs Predicted Values
plt.subplot(3, 1, 1)
plt.plot(np.arange(len(data)), data, label='Actual Values', color='blue')
plt.plot(np.arange(train_size, len(data)), arima_predictions, label='ARIMA Predictions', color='orange')
plt.plot(np.arange(train_size, len(data)), linear_predictions, label='Linear Regression Predictions', color='green')
plt.plot(X_test, nn_predictions, label='Neural Network Predictions', color='red')
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.grid(True)

# Plot Error Metrics
plt.subplot(3, 1, 2)
models = ['ARIMA', 'Linear Regression', 'Neural Network']
rmses = [arima_rmse, linear_rmse, nn_rmse]
maes = [arima_mae, linear_mae, nn_mae]
mapes = [arima_mape, linear_mape, nn_mape]

x = np.arange(len(models))

plt.bar(x - 0.2, rmses, 0.2, label='RMSE')
plt.bar(x, maes, 0.2, label='MAE')
plt.bar(x + 0.2, mapes, 0.2, label='MAPE')
plt.xlabel('Models')
plt.ylabel('Error Metrics')
plt.title('Error Metrics Comparison')
plt.xticks(x, models)
plt.legend()
plt.grid(True)

# Plot Detailed Error Analysis for ARIMA
plt.subplot(3, 1, 3)
errors = test - arima_predictions
plt.plot(errors, label='ARIMA Errors', marker='o')
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Error Values')
plt.title('ARIMA Prediction Errors')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
