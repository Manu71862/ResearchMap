import re
from typing import List, Dict, Any, Tuple, Optional
from rank_bm25 import BM25Okapi
from backend.config import Config

_WORD_REGEX = re.compile(r"\b[a-zA-Z0-9_\-]+\b")

def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    if not text:
        return []
    return [w.lower() for w in _WORD_REGEX.findall(text)]

class BM25Retriever:
    """
    Transparent lexical retrieval using BM25Okapi for scientific papers and sentences.
    Provides abstract-level retrieval from the corpus and sentence-level ranking.
    """

    def __init__(self, corpus: Dict[int, Dict[str, Any]]):
        self.corpus = corpus
        self.doc_ids: List[int] = []
        self.corpus_tokenized: List[List[str]] = []

        # Build document-level index
        for doc_id, doc_info in self.corpus.items():
            self.doc_ids.append(doc_id)
            title = doc_info.get("title", "")
            abstract_text = " ".join(doc_info.get("abstract", []))
            full_text = f"{title} {abstract_text}".strip()
            self.corpus_tokenized.append(tokenize(full_text))

        self.bm25 = BM25Okapi(self.corpus_tokenized) if self.corpus_tokenized else None

    def retrieve_documents(self, query: str, top_k: int = Config.BM25_TOP_K_DOCS) -> List[Dict[str, Any]]:
        """
        Retrieve top_k documents from the corpus for a given query string.
        Returns a list of dicts:
            [{"doc_id": int, "title": str, "abstract": List[str], "bm25_score": float}, ...]
        """
        if not self.bm25 or not query or not self.doc_ids:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        # Get top-k indices sorted descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0 and len(results) >= 1:
                # Discard non-matching documents if we already have candidates
                continue
            doc_id = self.doc_ids[idx]
            doc_data = self.corpus[doc_id]
            results.append({
                "doc_id": doc_id,
                "title": doc_data.get("title", ""),
                "abstract": doc_data.get("abstract", []),
                "bm25_score": score
            })

        return results

    def retrieve_sentences(
        self,
        query: str,
        candidate_docs: List[Dict[str, Any]],
        top_k_sentences: int = Config.BM25_TOP_K_SENTENCES,
        doc_score_weight: float = Config.BM25_DOC_SCORE_WEIGHT,
        max_per_doc: Optional[int] = Config.BM25_MAX_SENTENCES_PER_DOC,
    ) -> List[Dict[str, Any]]:
        """
        Rank individual sentences across the candidate documents for a claim.
        Combines sentence lexical match with parent document relevance prior
        so evidence in top-ranked documents is not displaced by distractor documents.
        Returns:
            [{"doc_id": int, "sentence_idx": int, "text": str, "bm25_score": float}, ...]
        """
        all_sentences: List[Dict[str, Any]] = []
        sentence_tokenized: List[List[str]] = []

        for doc in candidate_docs:
            doc_id = doc["doc_id"]
            doc_score = float(doc.get("bm25_score", 0.0))
            for s_idx, sent_text in enumerate(doc.get("abstract", [])):
                tokens = tokenize(sent_text)
                if not tokens:
                    continue
                all_sentences.append({
                    "doc_id": doc_id,
                    "sentence_idx": s_idx,
                    "text": sent_text,
                    "doc_bm25": doc_score,
                })
                sentence_tokenized.append(tokens)

        if not sentence_tokenized or not query:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        sent_bm25 = BM25Okapi(sentence_tokenized)
        scores = sent_bm25.get_scores(q_tokens)

        # Prioritize sentences using sentence score + scaled document relevance
        combined_scores = []
        for i, s_score in enumerate(scores):
            d_score = all_sentences[i]["doc_bm25"]
            combined = s_score + doc_score_weight * d_score
            combined_scores.append(combined)

        sorted_indices = sorted(range(len(combined_scores)), key=lambda i: combined_scores[i], reverse=True)

        results = []
        doc_counts: Dict[int, int] = {}
        for idx in sorted_indices:
            sent_info = all_sentences[idx].copy()
            did = sent_info["doc_id"]
            if max_per_doc and doc_counts.get(did, 0) >= max_per_doc:
                continue
            sent_info["bm25_score"] = float(scores[idx])
            sent_info.pop("doc_bm25", None)
            results.append(sent_info)
            doc_counts[did] = doc_counts.get(did, 0) + 1
            if len(results) >= top_k_sentences:
                break

        return results
