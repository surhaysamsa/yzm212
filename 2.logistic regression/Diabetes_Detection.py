import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv("diabetes.csv")
X = df.drop(columns = ["Outcome"])
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state = 42, stratify = y)
X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train] 
X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test] 
y_train = y_train.to_numpy().reshape(-1,1)
y_test = y_test.to_numpy().reshape(-1,1)
weights_initial = np.random.randn(X_train_bias.shape[1]) * 0.01
weights_initial = weights_initial.reshape(-1,1)


def sigmoid(Z):
    return 1 / (1 + np.exp(-1 * Z))
def compute_cost(X, y, W):
    Z = np.dot(X, W)
    y_pred = sigmoid(Z)
    cost = -1 * np.mean(y * np.log(y_pred) + (1 - y) * np.log(1- y_pred))
    return cost
def gradient_descent(X, y, W, learning_rate = 0.001, n_steps = 10000, print_cost = True):
    m = X.shape[0]
    for i in range(n_steps):
        Z = np.dot(X, W)
        y_pred = sigmoid(Z)
        gradient = np.dot(X.T , y_pred - y) / m 
        W -= learning_rate * gradient
        
        if print_cost and i % 100 == 0:
            print(f"Cost is {compute_cost(X, y, W)}")
    
    return W
weights_final = gradient_descent(X_train_bias, y_train, weights_initial, learning_rate = 0.0001, n_steps = 1000, print_cost = True)
y_pred_probs = sigmoid(np.dot(X_test_bias, weights_final))
y_preds = (y_pred_probs > 0.5).astype(int)
accuracy = np.mean(y_preds == y_test)
weights_initial = np.random.randn(X_train_bias.shape[1]) * 0.01
weights_initial = weights_initial.reshape(-1,1)
y_pred_probs = sigmoid(np.dot(X_test_bias, weights_initial))
y_preds = (y_pred_probs > 0.5).astype(int)
accuracy = np.mean(y_preds == y_test)