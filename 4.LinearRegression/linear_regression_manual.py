import numpy as np
import pandas as pd

# Load train/test splits
# Load train/test splits
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

# Ensure all features are numeric (handle categoricals if present)
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Align columns in test to train (in case dummies differ)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Convert to float
X_train = X_train.values.astype(float)
X_test = X_test.values.astype(float)
y_train = y_train.astype(float)
y_test = y_test.astype(float)

# Add bias term
X_train_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_b = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# Least squares solution: theta = (X^T X)^(-1) X^T y
theta = np.linalg.inv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train.reshape(-1, 1)

# Predictions
train_preds = X_train_b @ theta
mse_train = np.mean((train_preds.ravel() - y_train) ** 2)

test_preds = X_test_b @ theta
mse_test = np.mean((test_preds.ravel() - y_test) ** 2)

print(f'Manual Linear Regression (Least Squares)')
print(f'Train MSE: {mse_train:.4f}')
print(f'Test MSE: {mse_test:.4f}')

# Save predictions
np.savetxt('manual_train_preds.csv', train_preds, delimiter=',')
np.savetxt('manual_test_preds.csv', test_preds, delimiter=',')
