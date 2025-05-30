f# Naive Bayes Binary Classification Project

## Table of Contents
- [Problem Description](#problem-description)
- [Dataset](#dataset)
- [Method](#method)
- [Running the Project](#running-the-project)
    - [Clone the Repository](#clone-the-repository)
    - [Using a Virtual Environment (Preferred)](#using-a-virtual-environment-preferred-method)
    - [Direct Execution](#direct-execution-alternative-method)
- [Results](#results)
- [Discussion](#discussion)
    - [Performance Metrics Selection](#performance-metrics-selection)
    - [Implementation Comparison](#implementation-comparison)
- [Author Information](#author-information)

## Problem Description
This project focuses on implementing Naive Bayes for binary classification to determine the likelihood of diabetes in individuals based on a range of health metrics. By comparing a custom implementation with Scikit-learn's GaussianNB, we evaluate differences in accuracy and computational efficiency.

## Dataset
The dataset for this project is sourced from a comprehensive diabetes database, containing medical records of individuals. It includes features such as age, BMI, glucose levels, and blood pressure, which are crucial for predicting diabetes onset.

Key features:

- **Age**: Age of the individual
- **BMI**: Body mass index, a measure of body fat based on height and weight
- **Glucose Levels**: Blood glucose concentration
- **Blood Pressure**: Blood pressure readings
- **Additional Features**: Other relevant health metrics

The dataset comprises over 9500 samples with 17 features, providing a robust basis for model training and evaluation.

## Method
1. Scikit-learn's GaussianNB: Generates predictions saved to `scikit_predictions.csv`.
2. Custom implementation of Gaussian Naive Bayes: Generates predictions saved to `custom_predictions.csv`.
3. Comparison of models using `compareModels.py` to evaluate performance metrics.

## Running the Project
### Clone the Repository
```bash
# Clone the repository
git clone https://github.com/yourusername/yourrepository.git

# Go to Naive Bayes folder
cd yourrepository/naiveBayes
```

### Using a Virtual Environment (Preferred Method)
Using a virtual environment is the preferred way to run this project as it isolates dependencies and avoids conflicts with other projects.

#### On Windows:
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python runPipeline.py

# Deactivate the virtual environment when done
deactivate
```

#### On macOS and Linux:
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python runPipeline.py

# Deactivate the virtual environment when done
deactivate
```

### Direct Execution (Alternative Method)
If you prefer not to use a virtual environment, you can run the code directly:
```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python runPipeline.py
```

## Results
The performance of both models was evaluated using accuracy, precision, recall, and F1 score.

### Scikit-learn Implementation
- **Confusion Matrix**:
  - [[1892, 0], [109, 861]]
- **Accuracy**: 0.96
- **Precision**: 1.00
- **Recall**: 0.89
- **F1 Score**: 0.94
- **Training Time**: 0.0245 seconds
- **Prediction Time**: 0.0042 seconds

### Custom Implementation
- **Confusion Matrix**:
  - [[1892, 0], [110, 860]]
- **Accuracy**: 0.96
- **Precision**: 1.00
- **Recall**: 0.89
- **F1 Score**: 0.94
- **Training Time**: 0.0010 seconds
- **Prediction Time**: 0.0010 seconds

## Discussion
### Performance Metrics Selection
The performance metrics such as accuracy, precision, recall, and F1 score are crucial for evaluating the model's effectiveness, especially in medical diagnosis where false negatives can be critical. The balanced class distribution in the dataset ensures that accuracy is a reliable metric.

### Implementation Comparison
The custom implementation was faster in both training and prediction times, demonstrating the efficiency of a tailored approach. However, the Scikit-learn implementation offers robustness and ease of use, making it a valuable tool for quick prototyping and validation. The comparison confirmed that both models perform similarly in terms of accuracy and other metrics, with slight differences in the confusion matrix outcomes.

## Author Information
- **Name**: Mustafa Surhay Samsa
- **Student ID**: 22290159
- **Department**: Ankara Üniversitesi Yapay Zeka ve Veri Mühendisliği
