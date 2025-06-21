"""Train the scratch neural network on the diabetes dataset and create visualizations."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

from neural_network import NeuralNetwork
from preprocess import DATA_DIR, TRAIN_CSV, TEST_CSV, main as preprocess_main


def load_data():
    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        preprocess_main()
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    X_train = train_df.drop(columns=["Outcome"]).values
    y_train = train_df["Outcome"].values
    X_test = test_df.drop(columns=["Outcome"]).values
    y_test = test_df["Outcome"].values
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test


def main():
    X_train, y_train, X_test, y_test = load_data()
    n_features = X_train.shape[1]
    nn = NeuralNetwork([n_features, 16, 8, 1])
    losses = nn.fit(X_train, y_train, epochs=1000, lr=0.05, verbose=True)

    # Plot loss curve
    plt.figure()
    plt.plot(losses)
    plt.title("Binary Cross-Entropy Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(Path(DATA_DIR) / "loss_curve.png")

    # Evaluate
    y_pred = nn.predict(X_test).flatten()
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Confusion matrix heatmap
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(Path(DATA_DIR) / "confusion_matrix.png")

    # Save metrics
    with open(Path(DATA_DIR) / "metrics.json", "w") as fp:
        json.dump({"accuracy": acc, "classification_report": report}, fp, indent=2)

    print(f"Test Accuracy: {acc:.4f}")
    print("Detailed report saved to data/metrics.json")


if __name__ == "__main__":
    main()
