# Logistic Regression Binary Classification Project

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
This project aims to implement logistic regression for binary classification to predict the likelihood of diabetes in patients based on a variety of medical measurements. By comparing a custom implementation with Scikit-learn's logistic regression, we assess differences in performance and computational efficiency.

## Dataset
The dataset used in this project is derived from the Pima Indians Diabetes Database. It includes medical information from female patients of Pima Indian heritage, aimed at predicting diabetes onset based on various health indicators.

Key attributes:

- **Pregnancies**: Number of pregnancies
- **Glucose**: Plasma glucose concentration during an oral glucose tolerance test
- **Blood Pressure**: Diastolic blood pressure measured in mm Hg
- **Skin Thickness**: Triceps skinfold thickness in mm
- **Insulin**: Serum insulin levels after two hours (mu U/ml)
- **BMI**: Body mass index calculated as weight in kg divided by height in meters squared
- **Diabetes Pedigree Function**: A function that scores the likelihood of diabetes based on family history
- **Age**: Age of the patient in years
- **Outcome**: Binary indicator of diabetes (0 for non-diabetic, 1 for diabetic)

The dataset comprises 768 records with 8 features, and it is complete with no missing values.

## Method
1. Scikit-learn's LogisticRegression: Generates predictions and evaluates performance.
2. Custom implementation of Logistic Regression: Uses gradient descent for optimization.
3. Comparison of models using `ModelComparison.py` to evaluate performance metrics.

## Running the Project
### Clone the Repository
```bash
# Clone the repository
git clone https://github.com/yourusername/yourrepository.git

# Go to Logistic Regression folder
cd yourrepository/logistic_regression
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
python ModelComparison.py

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
python ModelComparison.py

# Deactivate the virtual environment when done
deactivate
```

### Direct Execution (Alternative Method)
If you prefer not to use a virtual environment, you can run the code directly:
```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python ModelComparison.py
```

## Results
The performance of both models was evaluated using accuracy and confusion matrices.

### Scikit-learn Implementation
- **Confusion Matrix**:
  - [[82, 18], [26, 28]]
- **Accuracy**: 0.7143
- **Training Time**: ~0.0222 seconds

### Custom Implementation
- **Confusion Matrix**:
  - [[36, 64], [8, 46]]
- **Accuracy**: 0.8200
- **Training Time**: ~0.022 seconds

## Discussion
### Performance Metrics Selection
The performance metrics such as accuracy and confusion matrices are crucial for evaluating the model's effectiveness, especially in medical diagnosis where false negatives can be critical. The balanced class distribution in the dataset ensures that accuracy is a reliable metric.

### Implementation Comparison
The custom implementation was slightly more accurate, demonstrating the effectiveness of a tailored approach. However, the Scikit-learn implementation offers robustness and ease of use, making it a valuable tool for quick prototyping and validation.

## Author Information
- **Name**: Mustafa Surhay Samsa
- **Student ID**: 22290159
- **Department**: Ankara Üniversitesi Yapay Zeka ve Veri Mühendisliği
