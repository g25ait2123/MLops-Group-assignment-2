"""Model selection and loading module.

Loads DistilBERT pre-trained model configured for book genre classification.
"""

import json

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def load_label_mapping(path="id2label.json"):
    """Load label mapping from id2label.json."""
    with open(path, "r") as f:
        id2label = json.load(f)
    label2id = {v: int(k) for k, v in id2label.items()}
    return id2label, label2id


def load_model_and_tokenizer(
    model_name="distilbert-base-uncased",
    id2label_path="id2label.json",
):
    """Load pre-trained model and tokenizer for sequence classification.

    Args:
        model_name: HuggingFace model identifier.
        id2label_path: Path to the id2label.json mapping file.

    Returns:
        Tuple of (model, tokenizer).
    """
    id2label, label2id = load_label_mapping(id2label_path)
    num_labels = len(id2label)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # Print model size info
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = total_params * 4 / (1024 * 1024)  # float32 = 4 bytes
    print(f"Model: {model_name}")
    print(f"Total parameters: {total_params:,}")
    print(f"Estimated size (float32): {size_mb:.1f} MB")
    print(f"Number of labels: {num_labels}")

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    print("\nModel loaded successfully!")
    print(f"Model config id2label: {model.config.id2label}")
