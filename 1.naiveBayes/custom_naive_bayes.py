import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import time

class CustomGaussianNB:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = np.array([X[y == c].mean(axis=0) for c in self.classes])
        self.var = np.array([X[y == c].var(axis=0) for c in self.classes]) + 1e-9  # Variance smoothing
        self.priors = np.array([np.mean(y == c) for c in self.classes])

    def predict(self, X):
        likelihoods = [self._calculate_likelihood(X, c) for c in range(len(self.classes))]
        return self.classes[np.argmax(likelihoods, axis=0)]

    def _calculate_likelihood(self, X, class_idx):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((X - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return np.prod(numerator / denominator, axis=1) * self.priors[class_idx]

# Load the dataset
df = pd.read_csv(r'C:\Users\samsa\Desktop\Naivebayesodev\diabetes_dataset.csv')

# Split the data into features and target variable
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the custom Gaussian Naive Bayes model
custom_gnb = CustomGaussianNB()

# Record the training start time
start_time = time.time()

# Train the model
custom_gnb.fit(X_train, y_train)

# Record the training end time
training_time = time.time() - start_time

# Record the prediction start time
start_time = time.time()

# Make predictions
y_pred = custom_gnb.predict(X_test)

# Record the prediction end time
prediction_time = time.time() - start_time

# Save predictions to CSV file
custom_predictions_df = pd.DataFrame({'true': y_test, 'predicted': y_pred})
custom_predictions_df.to_csv('custom_predictions.csv', index=False)

# Calculate performance metrics
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred, zero_division=1)

# Display the results
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'Training Time: {training_time:.4f} seconds')
print(f'Prediction Time: {prediction_time:.4f} seconds')
