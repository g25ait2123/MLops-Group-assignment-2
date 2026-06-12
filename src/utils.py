"""Shared utilities for the MLOps pipeline."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset


class GenreDataset(Dataset):
    """PyTorch Dataset for book genre classification."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(eval_pred):
    """Compute accuracy and weighted F1 for HF Trainer evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }
