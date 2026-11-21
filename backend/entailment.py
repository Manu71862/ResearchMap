import torch
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

from backend.data_loader import (
    LABEL_SUPPORTS,
    LABEL_CONTRADICTS,
    LABEL_NEI,
    CANONICAL_LABELS,
)

class NLIEntailmentModel:
    """
    Pretrained Natural Language Inference model wrapper for scientific claim verification.
    Accepts candidate evidence (premise) and scientific claim (hypothesis) pairs,
    producing calibrated probabilities for SUPPORTS, CONTRADICTS, and NOT ENOUGH INFORMATION.
    """

    DEFAULT_LABEL_MAPPING = {
        0: LABEL_CONTRADICTS,
        1: LABEL_SUPPORTS,
        2: LABEL_NEI,
    }
    LABEL_MAPPING = DEFAULT_LABEL_MAPPING

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small", device: str = "auto"):
        self.model_name = model_name
        self.device_str = device
        self.device = self._resolve_device(device)

        self.tokenizer = None
        self.model = None
        self.label_mapping = {}  # Maps model output index -> canonical label
        self._load_model()

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_model(self):
        """Load tokenizer and sequence classification model with dynamic id2label mapping."""
        try:
            print(f"Loading pretrained NLI model '{self.model_name}' on device '{self.device}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

            # Dynamically infer id2label mapping
            id2label = getattr(self.model.config, "id2label", {})
            for idx, raw_label in id2label.items():
                idx = int(idx)
                raw_upper = str(raw_label).upper()
                if "ENTAIL" in raw_upper or "SUPPORT" in raw_upper:
                    self.label_mapping[idx] = LABEL_SUPPORTS
                elif "CONTRA" in raw_upper:
                    self.label_mapping[idx] = LABEL_CONTRADICTS
                elif "NEUT" in raw_upper:
                    self.label_mapping[idx] = LABEL_NEI
                else:
                    self.label_mapping[idx] = LABEL_NEI

            # Fallback if id2label is missing
            if not self.label_mapping:
                # Default mapping for cross-encoder/nli-deberta-v3-small: 0: contra, 1: entail, 2: neutral
                self.label_mapping = {
                    0: LABEL_CONTRADICTS,
                    1: LABEL_SUPPORTS,
                    2: LABEL_NEI
                }
            print(f"Model loaded successfully. Label mapping: {self.label_mapping}")
        except Exception as e:
            print(f"Warning: Could not load model '{self.model_name}' from HuggingFace Hub: {e}")
            raise e

    def predict_pair(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        """
        Evaluate a single (premise=evidence, hypothesis=claim) pair.
        """
        results = self.predict_batch([(premise, hypothesis)])
        return results[0]

    def predict_batch(self, pairs: List[Tuple[str, str]], batch_size: int = 16) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of (premise, hypothesis) pairs.
        Returns list of dicts with probabilities, predicted label, and confidence score.
        """
        if not pairs:
            return []

        all_results = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]
            premises = [p[0] if p[0] else "" for p in batch_pairs]
            hypotheses = [p[1] if p[1] else "" for p in batch_pairs]

            encoded = self.tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1).cpu().numpy()

            for row_probs in probs:
                prob_dict = {
                    LABEL_SUPPORTS: 0.0,
                    LABEL_CONTRADICTS: 0.0,
                    LABEL_NEI: 0.0,
                }
                for idx, p in enumerate(row_probs):
                    canon_label = self.label_mapping.get(idx, LABEL_NEI)
                    prob_dict[canon_label] = float(p)

                # Determine predicted label and confidence
                pred_label = max(prob_dict.keys(), key=lambda k: prob_dict[k])
                confidence = prob_dict[pred_label]

                all_results.append({
                    "prediction": pred_label,
                    "confidence": confidence,
                    "probabilities": prob_dict,
                })

        return all_results
