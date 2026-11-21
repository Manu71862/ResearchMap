"""
Experimental Dense Retriever using sentence-transformers/all-MiniLM-L6-v2.

Features:
- Fast vector similarity search using cached document embeddings.
- On-the-fly candidate sentence encoding with CPU optimization.
- Matches interface of BM25Retriever for drop-in pipeline compatibility.
"""

import json
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class DenseRetriever:
    """
    Dense semantic retriever using all-MiniLM-L6-v2 embeddings.
    """
    def __init__(
        self,
        corpus: Dict[int, Dict[str, Any]],
        embeddings_path: Path,
        doc_ids_path: Path,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        self.corpus = corpus
        self.device = torch.device(device)
        self.model_id = model_id

        # Load embedding matrix and doc_ids
        if not embeddings_path.exists() or not doc_ids_path.exists():
            raise FileNotFoundError(f"Dense cache not found at {embeddings_path} or {doc_ids_path}")

        self.doc_matrix = np.load(embeddings_path).astype(np.float32)
        with open(doc_ids_path, "r", encoding="utf-8") as f:
            self.doc_ids = json.load(f)

        self.doc_id_to_idx = {did: idx for idx, did in enumerate(self.doc_ids)}

        # Load model for query and candidate sentence encoding
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    def encode_texts(self, texts: List[str], max_length: int = 128) -> np.ndarray:
        """Encode a batch of texts into normalized dense vectors."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        with torch.no_grad():
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(self.device)
            out = self.model(**encoded)
            emb = mean_pooling(out, encoded["attention_mask"])
            emb_norm = torch.nn.functional.normalize(emb, p=2, dim=1)
            return emb_norm.cpu().numpy().astype(np.float32)

    def retrieve_documents(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve top_k documents using cosine similarity with cached document embeddings.
        """
        if not query or self.doc_matrix.shape[0] == 0:
            return []

        q_emb = self.encode_texts([query], max_length=128)[0]  # shape (384,)
        # Fast dot product cosine similarity
        sim_scores = self.doc_matrix @ q_emb  # shape (N,)

        top_indices = np.argsort(-sim_scores)[:top_k]

        results = []
        for idx in top_indices:
            score = float(sim_scores[idx])
            did = self.doc_ids[idx]
            doc_data = self.corpus.get(did, {})
            results.append({
                "doc_id": did,
                "title": doc_data.get("title", ""),
                "abstract": doc_data.get("abstract", []),
                "dense_score": score,
                "bm25_score": score,  # compatibility fallback
            })
        return results

    def retrieve_sentences(
        self,
        query: str,
        candidate_docs: List[Dict[str, Any]],
        top_k_sentences: int = 4,
        doc_score_weight: float = 0.35,
        max_per_doc: Optional[int] = 2
    ) -> List[Dict[str, Any]]:
        """
        Rank individual sentences across candidate documents using dense cosine similarity.
        """
        all_sentences = []
        sentence_texts = []

        for doc in candidate_docs:
            doc_id = doc["doc_id"]
            d_score = float(doc.get("dense_score", doc.get("bm25_score", 0.0)))
            for s_idx, sent_text in enumerate(doc.get("abstract", [])):
                if not sent_text.strip():
                    continue
                all_sentences.append({
                    "doc_id": doc_id,
                    "sentence_idx": s_idx,
                    "text": sent_text,
                    "doc_score": d_score,
                })
                sentence_texts.append(sent_text)

        if not sentence_texts or not query:
            return []

        q_emb = self.encode_texts([query], max_length=128)[0]
        sent_embs = self.encode_texts(sentence_texts, max_length=128)
        s_scores = sent_embs @ q_emb  # shape (M,)

        combined_scores = []
        for i, s_score in enumerate(s_scores):
            d_score = all_sentences[i]["doc_score"]
            combined = float(s_score) + doc_score_weight * d_score
            combined_scores.append(combined)

        sorted_indices = sorted(range(len(combined_scores)), key=lambda i: combined_scores[i], reverse=True)

        results = []
        doc_counts: Dict[int, int] = {}
        for idx in sorted_indices:
            sent_info = all_sentences[idx].copy()
            did = sent_info["doc_id"]
            if max_per_doc and doc_counts.get(did, 0) >= max_per_doc:
                continue
            sent_info["dense_score"] = float(s_scores[idx])
            sent_info["bm25_score"] = float(s_scores[idx])  # compatibility
            sent_info.pop("doc_score", None)
            results.append(sent_info)
            doc_counts[did] = doc_counts.get(did, 0) + 1
            if len(results) >= top_k_sentences:
                break

        return results
