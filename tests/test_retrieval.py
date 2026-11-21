import pytest
from backend.retrieval import BM25Retriever, tokenize

def test_tokenize():
    tokens = tokenize("BRCA1 mutations increase breast cancer risk by 50%!")
    assert "brca1" in tokens
    assert "mutations" in tokens
    assert "increase" in tokens
    assert "breast" in tokens
    assert "cancer" in tokens

    # Empty text
    assert tokenize("") == []
    assert tokenize("   ") == []

def test_retrieval_ranking_and_empty():
    dummy_corpus = {
        101: {
            "doc_id": 101,
            "title": "Oncology Study on Breast Cancer and BRCA1",
            "abstract": [
                "BRCA1 gene mutations significantly correlate with high cancer risk.",
                "Patients were tracked over a ten year clinical period."
            ]
        },
        102: {
            "doc_id": 102,
            "title": "Quantum Computing and Qubits",
            "abstract": [
                "Superconducting qubits demonstrate coherence.",
                "Quantum gates provide exponential speedups."
            ]
        },
        103: {
            "doc_id": 103,
            "title": "Geology and Plate Tectonics",
            "abstract": ["Earthquake faults shift over geological epochs."]
        },
        104: {
            "doc_id": 104,
            "title": "Economics and Inflation",
            "abstract": ["Central banks regulate monetary policy and interest rates."]
        }
    }

    retriever = BM25Retriever(dummy_corpus)

    # Retrieval for cancer claim
    results = retriever.retrieve_documents("BRCA1 mutations increase cancer risk", top_k=2)
    assert len(results) > 0
    assert results[0]["doc_id"] == 101
    assert results[0]["bm25_score"] > 0

    # Retrieval for quantum query
    quantum_results = retriever.retrieve_documents("superconducting qubits quantum coherence", top_k=2)
    assert len(quantum_results) > 0
    assert quantum_results[0]["doc_id"] == 102

    # Empty query handling
    empty_results = retriever.retrieve_documents("", top_k=2)
    assert empty_results == []

    # Non-matching query
    irrelevant_results = retriever.retrieve_documents("xylophone accordion polka", top_k=2)
    assert len(irrelevant_results) == 0 or irrelevant_results[0]["bm25_score"] <= 0

def test_sentence_retrieval():
    dummy_corpus = {
        101: {
            "doc_id": 101,
            "title": "Diabetes Treatment",
            "abstract": [
                "Metformin reduces glucose levels in type 2 diabetes.",
                "Exercise also improves cardiovascular fitness."
            ]
        }
    }
    retriever = BM25Retriever(dummy_corpus)
    candidate_docs = retriever.retrieve_documents("metformin glucose", top_k=1)
    sentences = retriever.retrieve_sentences("metformin glucose", candidate_docs, top_k_sentences=1)

    assert len(sentences) == 1
    assert sentences[0]["doc_id"] == 101
    assert sentences[0]["sentence_idx"] == 0
    assert "Metformin reduces glucose" in sentences[0]["text"]
