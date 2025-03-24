import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import time


df = pd.read_csv('diabetes.csv')
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Custom Logistic Regression Model
from Diabetes_Detection import gradient_descent, sigmoid

# Prepare data for custom model
X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]
y_train = y_train.to_numpy().reshape(-1, 1)
y_test = y_test.to_numpy().reshape(-1, 1)
weights_initial = np.random.randn(X_train_bias.shape[1]) * 0.01
weights_initial = weights_initial.reshape(-1, 1)

# Train custom model
t_start_custom = time.time()
weights_final = gradient_descent(X_train_bias, y_train, weights_initial, learning_rate=0.001, n_steps=10000)
t_end_custom = time.time()

# Predict and evaluate custom model
y_pred_probs_custom = sigmoid(np.dot(X_test_bias, weights_final))
y_preds_custom = (y_pred_probs_custom > 0.5).astype(int)
accuracy_custom = np.mean(y_preds_custom == y_test)
conf_matrix_custom = confusion_matrix(y_test, y_preds_custom)

# Scikit-learn Logistic Regression Model
t_start_sklearn = time.time()
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train.ravel())
t_end_sklearn = time.time()

# Predict and evaluate Scikit-learn model
y_pred_sklearn = model.predict(X_test)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
conf_matrix_sklearn = confusion_matrix(y_test, y_pred_sklearn)

# Print results
print("Custom Logistic Regression Model:")
print(f"Training time: {t_end_custom - t_start_custom:.4f} seconds")
print(f"Accuracy: {accuracy_custom:.4f}")
print("Confusion Matrix:")
print(conf_matrix_custom)

print("\nScikit-learn Logistic Regression Model:")
print(f"Training time: {t_end_sklearn - t_start_sklearn:.4f} seconds")
print(f"Accuracy: {accuracy_sklearn:.4f}")
print("Confusion Matrix:")
print(conf_matrix_sklearn)

# Plot confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].matshow(conf_matrix_custom, cmap='coolwarm')
axes[0].set_title('Custom Model Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')

axes[1].matshow(conf_matrix_sklearn, cmap='coolwarm')
axes[1].set_title('Scikit-learn Model Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')

plt.show()
