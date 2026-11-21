"""
Build and cache dense document embeddings for the expanded 10,835-document corpus.

Model: sentence-transformers/all-MiniLM-L6-v2 (22.7M parameters, 384 dimensions)
Input: data/experimental/corpus_expanded_10k.jsonl
Output:
  - experiments/cache/doc_embeddings_minilm_10k.npy
  - experiments/cache/doc_ids_10k.json
  - experiments/cache/dense_index_metadata.json
"""

import os
import sys
import json
import time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EXP_CORPUS_PATH = BASE_DIR / "data" / "experimental" / "corpus_expanded_10k.jsonl"
CACHE_DIR = BASE_DIR / "experiments" / "cache"
EMB_PATH = CACHE_DIR / "doc_embeddings_minilm_10k.npy"
IDS_PATH = CACHE_DIR / "doc_ids_10k.json"
META_PATH = CACHE_DIR / "dense_index_metadata.json"

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64
MAX_LENGTH = 128

def mean_pooling(model_output, attention_mask):
    """Mean pooling taking attention mask into account."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def main():
    print("=" * 70)
    print("Phase 5A: Building Dense Document Embedding Cache (all-MiniLM-L6-v2)")
    print("=" * 70)

    if not EXP_CORPUS_PATH.exists():
        print(f"[Error] Expanded corpus not found at {EXP_CORPUS_PATH}")
        sys.exit(1)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load documents
    print(f"Loading corpus from: {EXP_CORPUS_PATH}")
    t0 = time.time()
    doc_ids = []
    texts_to_encode = []

    with open(EXP_CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            did = int(data["doc_id"])
            title = data.get("title", "").strip()
            abstract = data.get("abstract", [])
            # Combine title and lead sentences
            lead_abstract = " ".join(abstract[:3]).strip()
            full_text = f"{title}. {lead_abstract}".strip() if title else lead_abstract

            doc_ids.append(did)
            texts_to_encode.append(full_text)

    t_load = time.time() - t0
    n_docs = len(doc_ids)
    print(f"Loaded {n_docs} documents in {t_load:.2f}s.")

    # 2. Load model
    print(f"Loading dense model: {MODEL_ID} on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()

    # 3. Batch encode
    print(f"Encoding {n_docs} documents in batches of {BATCH_SIZE} on CPU...")
    t_start = time.time()
    embeddings_list = []

    for b_idx in range(0, n_docs, BATCH_SIZE):
        batch_texts = texts_to_encode[b_idx: b_idx + BATCH_SIZE]
        with torch.no_grad():
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )
            out = model(**encoded)
            emb = mean_pooling(out, encoded["attention_mask"])
            # Normalize for fast dot-product cosine similarity
            emb_norm = torch.nn.functional.normalize(emb, p=2, dim=1)
            embeddings_list.append(emb_norm.cpu().numpy().astype(np.float32))

        if (b_idx // BATCH_SIZE) % 25 == 0 or b_idx + BATCH_SIZE >= n_docs:
            done = min(b_idx + BATCH_SIZE, n_docs)
            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  Encoded {done}/{n_docs} documents ({rate:.1f} docs/sec)...", end="\r")

    t_encode = time.time() - t_start
    all_embeddings = np.vstack(embeddings_list)
    print(f"\nEncoding complete in {t_encode:.2f}s ({n_docs/t_encode:.1f} docs/sec).")
    print(f"Embedding matrix shape: {all_embeddings.shape}, dtype: {all_embeddings.dtype}")

    # 4. Save cache files
    print(f"Saving embeddings to: {EMB_PATH}")
    np.save(EMB_PATH, all_embeddings)
    emb_size_mb = EMB_PATH.stat().st_size / (1024 * 1024)
    print(f"Embeddings file size: {emb_size_mb:.2f} MB")

    print(f"Saving doc IDs to: {IDS_PATH}")
    with open(IDS_PATH, "w", encoding="utf-8") as f_out:
        json.dump(doc_ids, f_out)

    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": MODEL_ID,
        "embedding_dimension": int(all_embeddings.shape[1]),
        "document_count": n_docs,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "encoding_time_seconds": round(t_encode, 2),
        "docs_per_second": round(n_docs / t_encode, 1),
        "embedding_file_size_mb": round(emb_size_mb, 2),
        "embeddings_path": str(EMB_PATH),
        "doc_ids_path": str(IDS_PATH),
    }

    with open(META_PATH, "w", encoding="utf-8") as f_meta:
        json.dump(metadata, f_meta, indent=2)

    print(f"Saved metadata to: {META_PATH}")
    print("Dense cache construction complete!")

if __name__ == "__main__":
    main()
