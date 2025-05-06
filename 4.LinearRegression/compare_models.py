import pandas as pd
from sklearn.metrics import mean_squared_error

# Load true values
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

# Load predictions
manual_train_preds = pd.read_csv('manual_train_preds.csv', header=None).values.ravel()
manual_test_preds = pd.read_csv('manual_test_preds.csv', header=None).values.ravel()
sklearn_train_preds = pd.read_csv('sklearn_train_preds.csv').values.ravel()
sklearn_test_preds = pd.read_csv('sklearn_test_preds.csv').values.ravel()

# Calculate MSE
mse_manual_train = mean_squared_error(y_train, manual_train_preds)
mse_manual_test = mean_squared_error(y_test, manual_test_preds)
mse_sklearn_train = mean_squared_error(y_train, sklearn_train_preds)
mse_sklearn_test = mean_squared_error(y_test, sklearn_test_preds)

print('Model Comparison:')
print(f'Manual Least Squares - Train MSE: {mse_manual_train:.4f}, Test MSE: {mse_manual_test:.4f}')
print(f'Scikit-learn LinearRegression - Train MSE: {mse_sklearn_train:.4f}, Test MSE: {mse_sklearn_test:.4f}')

if mse_manual_test < mse_sklearn_test:
    print('Manual model performed better on test data.')
elif mse_manual_test > mse_sklearn_test:
    print('Scikit-learn model performed better on test data.')
else:
    print('Both models performed equally on test data.')
