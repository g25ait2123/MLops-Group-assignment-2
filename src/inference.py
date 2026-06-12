"""Inference script for book genre classification.

Loads a trained model from Hugging Face Hub and predicts genre for input text.
Usage: python src/inference.py "Your book description here"
"""

import os
import sys

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def predict(text, model_name=None):
    """Predict book genre from text description.

    Args:
        text: Book description string.
        model_name: HuggingFace model identifier. Defaults to env var or fallback.

    Returns:
        Predicted genre label string.
    """
    if model_name is None:
        model_name = os.environ.get(
            "HF_MODEL_NAME",
            "VikasVishwakarma22/distilbert-goodreads-pipeline",
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_id = torch.argmax(outputs.logits, dim=-1).item()
    id2label = model.config.id2label
    return id2label.get(str(predicted_id), id2label.get(predicted_id, f"Unknown ({predicted_id})"))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: No input text provided")
        print("Usage: python src/inference.py \"Your text here\"")
        sys.exit(1)

    input_text = sys.argv[1]
    if not input_text.strip():
        print("Error: Input text is empty")
        sys.exit(1)

    label = predict(input_text)
    print(f"Predicted genre: {label}")
