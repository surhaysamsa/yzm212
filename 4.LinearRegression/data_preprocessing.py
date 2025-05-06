import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('Walmart_Sales.csv')

# Handle missing values (simple forward fill, can be changed as needed)
df = df.fillna(method='ffill').dropna()

# Assume the last column is the target
target_col = df.columns[-1]
X = df.drop(target_col, axis=1)
y = df[target_col]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save splits
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print('Data preprocessing complete. Train/test splits saved.')
