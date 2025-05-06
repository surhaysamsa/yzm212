import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

mse_train = mean_squared_error(y_train, train_preds)
mse_test = mean_squared_error(y_test, test_preds)

print('Scikit-learn Linear Regression')
print(f'Train MSE: {mse_train:.4f}')
print(f'Test MSE: {mse_test:.4f}')

# Save predictions
pd.DataFrame(train_preds).to_csv('sklearn_train_preds.csv', index=False)
pd.DataFrame(test_preds).to_csv('sklearn_test_preds.csv', index=False)
