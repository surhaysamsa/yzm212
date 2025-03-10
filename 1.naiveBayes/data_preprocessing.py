import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv(r'C:\Users\samsa\Desktop\Naivebayesodev\diabetes_dataset.csv')

# Display basic information about the dataset
def display_basic_info(df):
    print("\nDataset Information:")
    print(df.info())
    print("\nFirst 5 Rows:")
    print(df.head())
    print("\nDescription:")
    print(df.describe())

# Check for missing values
def check_missing_values(df):
    print("\nMissing Values:")
    print(df.isnull().sum())

# Handle missing values
def handle_missing_values(df):
    # Fill numeric columns with mean
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    # No need to handle categorical columns as there are none with missing values

# Analyze class distribution
def analyze_class_distribution(df):
    print("\nClass Distribution:")
    print(df['Outcome'].value_counts())

# Execute data preprocessing steps
def main():
    display_basic_info(df)
    check_missing_values(df)
    handle_missing_values(df)
    analyze_class_distribution(df)

if __name__ == "__main__":
    main()
