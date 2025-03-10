# compareModels.py
# Script to compare the performance of Scikit-learn and custom Naive Bayes models

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load predictions
scikit_predictions = pd.read_csv('scikit_predictions.csv')
custom_predictions = pd.read_csv('custom_predictions.csv')

# Calculate performance metrics
metrics = ['accuracy', 'precision', 'recall', 'f1']
for metric in metrics:
    scikit_score = globals()[f'{metric}_score'](scikit_predictions['true'], scikit_predictions['predicted'])
    custom_score = globals()[f'{metric}_score'](custom_predictions['true'], custom_predictions['predicted'])
    print(f'Scikit-learn {metric.capitalize()}: {scikit_score:.2f}')
    print(f'Custom {metric.capitalize()}: {custom_score:.2f}')

# Compare confusion matrices
scikit_conf_matrix = confusion_matrix(scikit_predictions['true'], scikit_predictions['predicted'])
custom_conf_matrix = confusion_matrix(custom_predictions['true'], custom_predictions['predicted'])

print(f'Scikit-learn Confusion Matrix:\n{scikit_conf_matrix}')
print(f'Custom Confusion Matrix:\n{custom_conf_matrix}')
