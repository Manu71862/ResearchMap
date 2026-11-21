"""
Experimental Hybrid Retriever combining Lexical BM25 and Dense Semantic Embeddings.

Features:
- Reciprocal Rank Fusion (RRF) combining BM25 and Dense retrieval ranks.
- Sentence-level hybrid fusion combining lexical token matches with dense semantic similarity.
- Parent document hybrid prior to prioritize sentences from top consensus papers.
- Fully compatible with ClaimVerificationPipeline interface.
"""

from typing import Dict, List, Any, Optional, Set
from backend.retrieval import BM25Retriever
from scripts.experimental.dense_retriever import DenseRetriever

class HybridRetriever:
    """
    Hybrid retriever fusing BM25Okapi lexical matching and all-MiniLM-L6-v2 dense vectors.
    """
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        rrf_k: int = 60,
    ):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.corpus = bm25_retriever.corpus
        self.rrf_k = rrf_k

    def retrieve_documents(
        self,
        query: str,
        top_k: int = 10,
        candidate_pool: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieve candidate documents using Reciprocal Rank Fusion (RRF) of BM25 and Dense rankings.
        RRF(d) = 1 / (k + rank_bm25(d)) + 1 / (k + rank_dense(d))
        """
        if not query:
            return []

        # Step 1: Lexical BM25 candidate list
        bm25_docs = self.bm25.retrieve_documents(query, top_k=candidate_pool)
        # Step 2: Dense semantic candidate list
        dense_docs = self.dense.retrieve_documents(query, top_k=candidate_pool)

        # Map doc_id to ranks and scores
        bm25_ranks = {d["doc_id"]: rank for rank, d in enumerate(bm25_docs, start=1)}
        dense_ranks = {d["doc_id"]: rank for rank, d in enumerate(dense_docs, start=1)}

        all_doc_ids = set(bm25_ranks.keys()) | set(dense_ranks.keys())
        rrf_scores: Dict[int, float] = {}

        for did in all_doc_ids:
            score = 0.0
            if did in bm25_ranks:
                score += 1.0 / (self.rrf_k + bm25_ranks[did])
            if did in dense_ranks:
                score += 1.0 / (self.rrf_k + dense_ranks[did])
            rrf_scores[did] = score

        # Sort descending by RRF score
        sorted_dids = sorted(all_doc_ids, key=lambda did: rrf_scores[did], reverse=True)[:top_k]

        results = []
        for did in sorted_dids:
            doc_data = self.corpus.get(did, {})
            # Look up raw scores
            b_score = next((d["bm25_score"] for d in bm25_docs if d["doc_id"] == did), 0.0)
            d_score = next((d.get("dense_score", 0.0) for d in dense_docs if d["doc_id"] == did), 0.0)

            results.append({
                "doc_id": did,
                "title": doc_data.get("title", ""),
                "abstract": doc_data.get("abstract", []),
                "bm25_score": float(b_score),
                "dense_score": float(d_score),
                "rrf_score": float(rrf_scores[did]),
                "combined_score": float(rrf_scores[did]),
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
        Rank candidate sentences across candidate documents using Hybrid RRF fusion.
        Combines sentence lexical rank, dense sentence similarity rank, and document RRF prior.
        """
        if not candidate_docs or not query:
            return []

        # Retrieve sentence rankings from both retrievers across the same candidate docs
        # Note: we ask for a large pool to calculate ranks
        bm25_sents = self.bm25.retrieve_sentences(
            query,
            candidate_docs,
            top_k_sentences=len(candidate_docs) * 10,
            doc_score_weight=0.0,
            max_per_doc=None
        )
        dense_sents = self.dense.retrieve_sentences(
            query,
            candidate_docs,
            top_k_sentences=len(candidate_docs) * 10,
            doc_score_weight=0.0,
            max_per_doc=None
        )

        bm25_sent_ranks = {(s["doc_id"], s["sentence_idx"]): rank for rank, s in enumerate(bm25_sents, start=1)}
        dense_sent_ranks = {(s["doc_id"], s["sentence_idx"]): rank for rank, s in enumerate(dense_sents, start=1)}

        all_keys = set(bm25_sent_ranks.keys()) | set(dense_sent_ranks.keys())
        doc_prior_map = {d["doc_id"]: d.get("rrf_score", d.get("bm25_score", 0.0)) for d in candidate_docs}

        # Normalize doc priors
        max_doc_prior = max(doc_prior_map.values()) if doc_prior_map else 1.0
        norm_doc_prior = {did: (val / max_doc_prior if max_doc_prior > 0 else 0.0) for did, val in doc_prior_map.items()}

        sentence_info_map = {}
        for s in bm25_sents + dense_sents:
            key = (s["doc_id"], s["sentence_idx"])
            if key not in sentence_info_map:
                sentence_info_map[key] = {
                    "doc_id": s["doc_id"],
                    "sentence_idx": s["sentence_idx"],
                    "text": s["text"],
                }

        combined_scores: Dict[tuple, float] = {}
        for key in all_keys:
            did, sidx = key
            s_rrf = 0.0
            if key in bm25_sent_ranks:
                s_rrf += 1.0 / (self.rrf_k + bm25_sent_ranks[key])
            if key in dense_sent_ranks:
                s_rrf += 1.0 / (self.rrf_k + dense_sent_ranks[key])

            d_prior = norm_doc_prior.get(did, 0.0)
            combined = s_rrf + doc_score_weight * d_prior * (1.0 / self.rrf_k)
            combined_scores[key] = combined

        sorted_keys = sorted(all_keys, key=lambda k: combined_scores[k], reverse=True)

        results = []
        doc_counts: Dict[int, int] = {}
        for key in sorted_keys:
            did, sidx = key
            if max_per_doc and doc_counts.get(did, 0) >= max_per_doc:
                continue
            item = sentence_info_map[key].copy()
            item["hybrid_score"] = float(combined_scores[key])
            item["bm25_score"] = float(combined_scores[key])  # compatibility with pipeline
            results.append(item)
            doc_counts[did] = doc_counts.get(did, 0) + 1
            if len(results) >= top_k_sentences:
                break

        return results
