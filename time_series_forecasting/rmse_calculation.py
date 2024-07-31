import numpy as np
import matplotlib.pyplot as plt

# Function to calculate RMSE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

# Function to calculate MAE
def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Function to calculate MAPE
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Actual and predicted values
Y_actual = np.array([100, 102, 101, 103, 105])
Y_predicted = np.array([98, 101, 102, 104, 106])

# Calculate RMSE
rmse = calculate_rmse(Y_actual, Y_predicted)
print(f'RMSE: {rmse}')

# Calculate MAE
mae = calculate_mae(Y_actual, Y_predicted)
print(f'MAE: {mae}')

# Calculate MAPE
mape = calculate_mape(Y_actual, Y_predicted)
print(f'MAPE: {mape:.2f}%')

# Plot actual vs predicted values
plt.figure(figsize=(10, 5))
plt.plot(Y_actual, label='Actual Values', marker='o')
plt.plot(Y_predicted, label='Predicted Values', marker='x')
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.grid(True)
plt.show()

# Plot error values
errors = Y_actual - Y_predicted
plt.figure(figsize=(10, 5))
plt.plot(errors, label='Errors', marker='s')
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Error Values')
plt.title('Prediction Errors')
plt.legend()
plt.grid(True)
plt.show()

# Print detailed error analysis
print("Detailed Error Analysis:")
for i, (actual, predicted, error) in enumerate(zip(Y_actual, Y_predicted, errors)):
    print(f'Time {i+1}: Actual = {actual}, Predicted = {predicted}, Error = {error}')
