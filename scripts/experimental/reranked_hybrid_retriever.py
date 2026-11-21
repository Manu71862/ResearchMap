"""
Approach D: Two-Stage Hybrid Retrieval with Lightweight Cross-Alignment Reranking.

Stage 1: Broad candidate document retrieval using Hybrid RRF (candidate pool = 25).
Stage 2: Semantic cross-alignment reranking using fine-grained dense query-sentence max-pooling
         and lexical density to select the final top 10 documents and top 4 diverse evidence sentences.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from scripts.experimental.hybrid_retriever import HybridRetriever

class RerankedHybridRetriever(HybridRetriever):
    """
    Two-stage retrieval: broad hybrid pool generation followed by dense-lexical reranking.
    """
    def __init__(
        self,
        bm25_retriever,
        dense_retriever,
        candidate_pool: int = 25,
        rrf_k: int = 60,
    ):
        super().__init__(bm25_retriever, dense_retriever, rrf_k=rrf_k)
        self.candidate_pool = candidate_pool

    def retrieve_documents(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Stage 1: Retrieve candidate_pool (25) documents using Hybrid RRF.
        Stage 2: Rerank using dense query-sentence max-alignment + normalized BM25.
        """
        initial_candidates = super().retrieve_documents(query, top_k=self.candidate_pool, candidate_pool=50)
        if not initial_candidates:
            return []

        # Encode query
        q_emb = self.dense.encode_texts([query], max_length=128)[0]

        # Rerank candidates based on max sentence alignment + normalized BM25
        reranked = []
        max_b_score = max((d["bm25_score"] for d in initial_candidates), default=1.0)
        norm_b = lambda b: (b / max_b_score) if max_b_score > 0 else 0.0

        for doc in initial_candidates:
            sents = doc.get("abstract", [])
            if not sents:
                max_sent_sim = doc.get("dense_score", 0.0)
            else:
                sent_embs = self.dense.encode_texts(sents[:6], max_length=128)
                sent_sims = sent_embs @ q_emb
                max_sent_sim = float(np.max(sent_sims)) if len(sent_sims) > 0 else 0.0

            # Fused reranking score: 55% dense sentence max-alignment + 45% normalized BM25
            rerank_score = 0.55 * max_sent_sim + 0.45 * norm_b(doc["bm25_score"])

            doc_copy = doc.copy()
            doc_copy["rerank_score"] = rerank_score
            doc_copy["bm25_score"] = rerank_score  # pipeline compatibility
            reranked.append(doc_copy)

        # Sort by rerank_score descending
        reranked.sort(key=lambda d: d["rerank_score"], reverse=True)
        return reranked[:top_k]
