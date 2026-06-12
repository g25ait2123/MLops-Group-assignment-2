"""Data preparation module for book genre classification.

Loads Goodreads dataset, inspects, cleans, encodes labels,
performs stratified train-test split, and saves processed data.
"""

import json
import os
import string

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


# Genre names from the dataset
GENRE_NAMES = [
    "History & Politics",
    "Health & Medicine",
    "Mystery & Thriller",
    "Arts & Design",
    "Self-Help & Wellness",
    "Sports & Recreation",
    "Non-Fiction",
    "Science Fiction & Fantasy",
    "Countries & Geography",
    "Other",
    "Nature & Environment",
    "Business & Finance",
    "Romance",
    "Philosophy & Religion",
    "Literature & Fiction",
    "Science & Technology",
    "Children & Young Adult",
    "Food & Cooking",
]


def load_goodreads_data():
    """Load Goodreads book genre dataset from Hugging Face Datasets hub."""
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("pszemraj/goodreads-bookgenres", split="train")
    df = pd.DataFrame(dataset)
    print(f"Loaded {len(df)} samples")
    return df


def inspect_data(df):
    """Print dataset size, column structure, class distribution, and quality."""
    print("\n=== Data Inspection ===")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nMissing values:\n{df[['Book', 'Description']].isnull().sum()}")
    print(f"\nGenre names ({len(GENRE_NAMES)}): {GENRE_NAMES}")


def convert_to_single_label(df):
    """Convert multi-label (list of 0/1) to single-label using first active genre."""
    def get_first_genre(genres_list):
        for idx, val in enumerate(genres_list):
            if val == 1:
                return GENRE_NAMES[idx]
        return None

    df["genre"] = df["Genres"].apply(get_first_genre)
    df = df.dropna(subset=["genre"])
    print(f"\nClass distribution:\n{df['genre'].value_counts()}")
    return df


def clean_text(df):
    """Clean text: lowercase, strip punctuation, remove duplicates, handle NAs."""
    print("\n=== Cleaning Text ===")

    # Handle missing values
    df = df.dropna(subset=["Description"])

    # Lowercase
    df = df.copy()
    df["Description"] = df["Description"].str.lower()

    # Strip punctuation
    translator = str.maketrans("", "", string.punctuation)
    df["Description"] = df["Description"].str.translate(translator)

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["Description"])
    print(f"Removed {before - len(df)} duplicates")

    print(f"Final dataset size: {len(df)}")
    return df


def encode_labels(df):
    """Encode class labels as integers and save id2label.json."""
    sorted_labels = sorted(df["genre"].unique())
    label2id = {label: idx for idx, label in enumerate(sorted_labels)}
    id2label = {str(idx): label for idx, label in enumerate(sorted_labels)}

    df = df.copy()
    df["label"] = df["genre"].map(label2id)

    # Save id2label.json
    with open("id2label.json", "w") as f:
        json.dump(id2label, f, indent=2)

    print(f"\nEncoded {len(sorted_labels)} labels")
    print(f"Saved id2label.json")
    return df, label2id, id2label


def stratified_split(df, test_size=0.2, random_state=42):
    """Perform stratified train-test split."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )
    print(f"\nTrain size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    return train_df, test_df


def save_datasets(train_df, test_df):
    """Save processed datasets as CSV files."""
    os.makedirs("data", exist_ok=True)
    train_df[["Description", "label"]].to_csv("data/train.csv", index=False)
    test_df[["Description", "label"]].to_csv("data/test.csv", index=False)
    print("\nSaved data/train.csv and data/test.csv")


def main():
    """Run the complete data preparation pipeline."""
    # Load
    df = load_goodreads_data()

    # Inspect
    inspect_data(df)

    # Convert multi-label to single-label
    df = convert_to_single_label(df)

    # Clean
    df = clean_text(df)

    # Encode
    df, label2id, id2label = encode_labels(df)

    # Split
    train_df, test_df = stratified_split(df)

    # Save
    save_datasets(train_df, test_df)

    print("\n=== Data preparation complete ===")
    print(f"id2label.json saved with {len(id2label)} classes")


if __name__ == "__main__":
    main()
