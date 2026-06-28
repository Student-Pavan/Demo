from __future__ import annotations

import argparse
import os
import pickle
from typing import List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ALLOWED_LABELS = {"easy", "medium", "hard"}


def _fallback_samples() -> Tuple[List[str], List[str]]:
    texts = [
        "What is the capital of France?",
        "Define photosynthesis in one sentence.",
        "Name the largest planet in the solar system.",
        "Explain how binary search works and analyze its time complexity.",
        "What is the derivative of x squared?",
        "Describe the causes and outcomes of World War I.",
        "Prove that the sum of first n natural numbers is n times n plus one by two.",
        "Compare mitosis and meiosis with key differences.",
        "Discuss transformers in NLP and the role of self-attention.",
    ]
    labels = [
        "easy",
        "easy",
        "easy",
        "medium",
        "medium",
        "medium",
        "hard",
        "hard",
        "hard",
    ]
    return texts, labels


def load_data(csv_path: str | None) -> Tuple[List[str], List[str]]:
    if not csv_path:
        return _fallback_samples()

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    data = pd.read_csv(csv_path)
    required_columns = {"question", "label"}
    if not required_columns.issubset(data.columns):
        raise ValueError("CSV must contain columns: question,label")

    cleaned = data.dropna(subset=["question", "label"]).copy()
    cleaned["label"] = cleaned["label"].astype(str).str.strip().str.lower()
    cleaned = cleaned[cleaned["label"].isin(ALLOWED_LABELS)]

    if cleaned.empty:
        raise ValueError("No valid rows found after filtering labels to easy/medium/hard.")

    texts = cleaned["question"].astype(str).tolist()
    labels = cleaned["label"].tolist()
    return texts, labels


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    lowercase=True,
                    strip_accents="unicode",
                    min_df=1,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train_and_save(csv_path: str | None, output_path: str, test_size: float, random_state: int) -> None:
    texts, labels = load_data(csv_path)

    unique_labels = sorted(set(labels))
    min_fraction_for_stratify = len(unique_labels) / max(len(labels), 1)
    adjusted_test_size = max(test_size, min_fraction_for_stratify)
    adjusted_test_size = min(adjusted_test_size, 0.5)

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=adjusted_test_size,
        random_state=random_state,
        stratify=labels,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    print("Classification report:")
    print(classification_report(y_test, predictions, labels=["easy", "medium", "hard"]))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as model_file:
        pickle.dump(pipeline, model_file)

    print(f"Model saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train TF-IDF + Logistic Regression model for question difficulty classification."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV path with columns: question,label (labels: easy, medium, hard)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/difficulty_model.pkl",
        help="Output pickle file path",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()
    train_and_save(
        csv_path=args.csv,
        output_path=args.output,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
