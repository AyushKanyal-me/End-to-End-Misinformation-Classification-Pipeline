"""
data_loader.py — Download and load the WELFake dataset.

The WELFake dataset contains 72,134 news articles labeled as
real (1) or fake (0). It merges data from Kaggle, McIntire,
Reuters, and BuzzFeed Political sources.

Source: https://zenodo.org/record/4561253

Run independently:
    python3 src/data_loader.py

This will download the dataset and save a processed version
to data/processed.pkl for use by other modules.
"""

import os
import pickle
import requests
import pandas as pd

# Direct download link from Zenodo
DATASET_URL = "https://zenodo.org/record/4561253/files/WELFake_Dataset.csv?download=1"

# Paths (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_PATH = os.path.join(DATA_DIR, "WELFake_Dataset.csv")
PROCESSED_PATH = os.path.join(DATA_DIR, "processed.pkl")


def download_dataset():
    """
    Download the WELFake dataset from Zenodo if it doesn't already exist locally.
    Shows a progress indicator during download.
    """
    if os.path.exists(DATASET_PATH):
        print(f"✓ Dataset already exists at {DATASET_PATH}")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading WELFake dataset from Zenodo...")
    print(f"  URL: {DATASET_URL}")

    response = requests.get(DATASET_URL, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(DATASET_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = (downloaded / total_size) * 100
                print(f"\r  Progress: {pct:.1f}% ({downloaded // (1024*1024)} MB)", end="", flush=True)

    print(f"\n✓ Dataset downloaded to {DATASET_PATH}")


def load_dataset(force_reload=False):
    """
    Load the WELFake dataset into a pandas DataFrame.

    If a processed version exists on disk, loads that directly.
    Otherwise, downloads, processes, and saves.

    Processing steps:
    1. Download if not cached
    2. Drop the unnamed index column
    3. Drop rows with missing title or text
    4. Combine title + text into a single 'content' column
    5. Save processed DataFrame to data/processed.pkl

    Args:
        force_reload: if True, re-process from CSV even if processed.pkl exists

    Returns:
        pd.DataFrame with columns 'content' (str) and 'label' (int: 0=fake, 1=real)
    """
    # Check for cached processed data
    if os.path.exists(PROCESSED_PATH) and not force_reload:
        print(f"✓ Loading processed data from {PROCESSED_PATH}")
        df = pd.read_pickle(PROCESSED_PATH)
        print(f"  Shape: {df.shape}")
        print(f"  Real (1): {(df['label'] == 1).sum():,}  |  Fake (0): {(df['label'] == 0).sum():,}")
        return df

    # Step 1: Ensure dataset is downloaded
    download_dataset()

    # Step 2: Load the CSV
    print("Loading dataset into memory...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  Raw shape: {df.shape}")

    # Step 3: Drop the unnamed index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Step 4: Drop rows with missing title or text
    initial_len = len(df)
    df = df.dropna(subset=["title", "text"])
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with missing title/text")

    # Step 5: Combine title + text into a single content column
    df["content"] = df["title"].astype(str) + " " + df["text"].astype(str)
    df = df[["content", "label"]]

    # Step 6: Ensure label is integer
    df["label"] = df["label"].astype(int)

    # Step 7: Save processed data for future use
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_pickle(PROCESSED_PATH)

    print(f"  Final shape: {df.shape}")
    print(f"  Label distribution:")
    print(f"    Real (1): {(df['label'] == 1).sum():,}")
    print(f"    Fake (0): {(df['label'] == 0).sum():,}")
    print(f"✓ Dataset loaded and saved to {PROCESSED_PATH}")

    return df


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1: Download & Load Dataset")
    print("=" * 60)
    df = load_dataset()
    print("\nSample rows:")
    print(df.head())
    print(f"\n✓ Done! Processed data saved to data/processed.pkl")
    print(f"  Next step: python3 src/train.py --model <model_name>")
