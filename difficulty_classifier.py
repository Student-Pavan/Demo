from __future__ import annotations

import os
import pickle
from typing import Dict, List


class DifficultyClassifier:
    """Classify question difficulty using a trained sklearn model."""

    def __init__(
        self,
        model_path: str = "models/difficulty_model.pkl",
    ) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Sklearn model not found at '{model_path}'. Provide a trained model file."
            )

        with open(model_path, "rb") as model_file:
            self.model = pickle.load(model_file)

    def classify(self, question: str) -> Dict[str, object]:
        """Predict difficulty for one question."""
        text_batch = [question]
        prediction = self.model.predict(text_batch)[0]
        result: Dict[str, object] = {"difficulty": str(prediction)}

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(text_batch)[0]
            confidence = float(max(proba))
            result["confidence"] = round(confidence, 4)

        return result

    def classify_batch(self, questions: List[str]) -> List[Dict[str, object]]:
        """Predict difficulty for a list of questions."""
        return [self.classify(question) for question in questions]
