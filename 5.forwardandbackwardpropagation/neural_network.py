"""Simple feedforward neural network implemented from scratch using NumPy.
Supports multiple hidden layers with sigmoid activation and trains via
batch gradient descent with backpropagation.
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    # derivative w.r.t pre-activation z where a = sigmoid(z)
    return a * (1 - a)


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


class NeuralNetwork:
    """Feed-forward neural network for binary classification."""

    def __init__(self, layer_sizes: List[int], seed: int = 42):
        """layer_sizes: list like [n_features, 16, 8, 1]"""
        rng = np.random.default_rng(seed)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            # Xavier initialization
            limit = np.sqrt(6 / (in_size + out_size))
            self.weights.append(rng.uniform(-limit, limit, size=(in_size, out_size)))
            self.biases.append(np.zeros((1, out_size)))

    # ============================================================
    # Forward / backward propagation
    # ============================================================
    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Returns (zs, activations) where activations[0] is X"""
        activations = [X]
        zs = []
        a = X
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            z = a @ W + b
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)
        # output layer
        z = a @ self.weights[-1] + self.biases[-1]
        zs.append(z)
        a = sigmoid(z)
        activations.append(a)
        return zs, activations

    def _backward(self, y_true: np.ndarray, zs: List[np.ndarray], activations: List[np.ndarray]):
        m = y_true.shape[0]
        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        # Output layer error
        delta = activations[-1] - y_true  # derivative of BCE with sigmoid output
        grads_w[-1] = (activations[-2].T @ delta) / m
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True) / m

        # Backpropagate through hidden layers
        for l in range(2, len(self.weights) + 1):
            z = zs[-l]
            sp = sigmoid_derivative(activations[-l])
            delta = (delta @ self.weights[-l + 1].T) * sp
            grads_w[-l] = (activations[-l - 1].T @ delta) / m
            grads_b[-l] = np.sum(delta, axis=0, keepdims=True) / m
        return grads_w, grads_b

    # ============================================================
    # Training API
    # ============================================================
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 500,
        lr: float = 0.01,
        verbose: bool = True,
    ) -> List[float]:
        y = y.reshape(-1, 1)
        losses: List[float] = []
        for epoch in range(1, epochs + 1):
            zs, activations = self._forward(X)
            loss = binary_cross_entropy(y, activations[-1])
            losses.append(loss)

            grads_w, grads_b = self._backward(y, zs, activations)
            # Gradient descent update
            for i in range(len(self.weights)):
                self.weights[i] -= lr * grads_w[i]
                self.biases[i] -= lr * grads_b[i]

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs} - loss: {loss:.4f}")
        return losses

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, activations = self._forward(X)
        return activations[-1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
