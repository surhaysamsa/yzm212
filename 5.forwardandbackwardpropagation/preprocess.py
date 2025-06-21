"""Data preprocessing for Pima Indians Diabetes dataset.
Downloads the dataset (if not present), cleans it, and splits into train / test CSVs.
"""
import os
from pathlib import Path
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split

# Constants
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_CSV = DATA_DIR / "diabetes.csv"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
)

def download_dataset():
    """Downloads the raw CSV if it does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_CSV.exists():
        print("Downloading dataset ...")
        urllib.request.urlretrieve(DATA_URL, RAW_CSV)
        print(f"Dataset saved to {RAW_CSV}")
    else:
        print("Dataset already exists. Skipping download.")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning. Replace zero values in certain columns considered missing."""
    cols_with_missing = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]
    for col in cols_with_missing:
        df[col].replace(0, pd.NA, inplace=True)
        df[col].fillna(df[col].median(), inplace=True)
    return df


def split_and_save(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["Outcome"])
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    print(f"Saved train set to {TRAIN_CSV} (n={len(train_df)})")
    print(f"Saved test set to {TEST_CSV} (n={len(test_df)})")


def main():
    download_dataset()
    df = pd.read_csv(RAW_CSV)
    df = clean_data(df)
    split_and_save(df)


if __name__ == "__main__":
    main()
