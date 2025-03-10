import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time

# Load the dataset
df = pd.read_csv(r'C:\Users\samsa\Desktop\Naivebayesodev\diabetes_dataset.csv')

# Split the data into features and target variable
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Gaussian Naive Bayes model
gnb = GaussianNB()

# Record the training start time
start_time = time.time()

# Train the model
gnb.fit(X_train, y_train)

# Record the training end time
training_time = time.time() - start_time

# Record the prediction start time
start_time = time.time()

# Make predictions
y_pred = gnb.predict(X_test)

# Record the prediction end time
prediction_time = time.time() - start_time

# Calculate performance metrics
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Save predictions to CSV file
scikit_predictions_df = pd.DataFrame({'true': y_test, 'predicted': y_pred})
scikit_predictions_df.to_csv('scikit_predictions.csv', index=False)

# Display the results
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'Training Time: {training_time:.4f} seconds')
print(f'Prediction Time: {prediction_time:.4f} seconds')
