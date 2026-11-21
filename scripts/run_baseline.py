"""
Script to run the verification pipeline on an arbitrary scientific claim from the CLI.
Accepts a claim string, retrieves relevant literature abstracts via BM25,
evaluates candidate sentences using the fine-tuned (or baseline) NLI model,
and outputs the predicted label, confidence, and preserved source evidence.
"""

import sys
import argparse
import json
from pathlib import Path

# Ensure workspace root is in python path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.config import Config
from backend.data_loader import load_corpus
from backend.retrieval import BM25Retriever
from backend.entailment import NLIEntailmentModel
from backend.pipeline import ClaimVerificationPipeline

def main():
    parser = argparse.ArgumentParser(description="Verify a scientific claim using ResearchMap pipeline.")
    parser.add_argument("--claim", type=str, required=True, help="Scientific claim text to verify")
    parser.add_argument("--top-k-docs", type=int, default=Config.BM25_TOP_K_DOCS, help="BM25 top-k documents")
    parser.add_argument("--top-k-sentences", type=int, default=Config.BM25_TOP_K_SENTENCES, help="BM25 top-k sentences")
    parser.add_argument("--model-path", type=str, default=None, help="Path to NLI model checkpoint")
    parser.add_argument("--json", action="store_true", help="Output full JSON result")

    args = parser.parse_args()

    corpus_path = Config.RAW_DATA_DIR / "corpus.jsonl"
    if not corpus_path.exists():
        print(f"Error: Corpus file not found at {corpus_path}. Run scripts/download_scifact.py first.")
        sys.exit(1)

    print(f"Loading corpus from {corpus_path}...")
    corpus = load_corpus(corpus_path)
    print(f"Loaded {len(corpus):,} papers.")

    retriever = BM25Retriever(corpus)

    finetuned_dir = Config.MODELS_DIR / "finetuned_nli"
    if args.model_path:
        model_name = args.model_path
    elif finetuned_dir.exists() and (finetuned_dir / "model.safetensors").exists():
        model_name = str(finetuned_dir)
        print(f"Using fine-tuned checkpoint: {model_name}")
    else:
        model_name = Config.NLI_MODEL_NAME
        print(f"Using baseline model: {model_name}")

    entailment_model = NLIEntailmentModel(model_name=model_name, device=Config.DEVICE)
    pipeline = ClaimVerificationPipeline(
        retriever=retriever,
        entailment_model=entailment_model,
        top_k_docs=args.top_k_docs,
        top_k_sentences=args.top_k_sentences,
    )

    print(f"\nEvaluating Claim: \"{args.claim}\"")
    result = pipeline.verify_claim(args.claim)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("\n" + "=" * 60)
    print("VERIFICATION RESULT")
    print("=" * 60)
    print(f"Prediction: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Candidate Documents Retrieved: {result['retrieved_candidate_doc_ids']}")

    if result["evidence_details"]:
        print("\nCandidate Evidence Evaluated:")
        for idx, ev in enumerate(result["evidence_details"], 1):
            print(f"\n[{idx}] Document {ev['doc_id']} ('{ev['title']}'):")
            print(f"    Sentence #{ev['sentence_idx']}: \"{ev['text']}\"")
            print(f"    BM25 Score: {ev['bm25_score']:.2f}")
            print(f"    NLI Label:  {ev['entailment_label']} (Conf: {ev['confidence']:.3f})")
            print(f"    Probabilities: SUPPORTS={ev['probabilities']['SUPPORTS']:.3f}, CONTRADICTS={ev['probabilities']['CONTRADICTS']:.3f}, NEI={ev['probabilities']['NOT ENOUGH INFORMATION']:.3f}")
    else:
        print("\nNo candidate evidence found.")
    print("=" * 60)

if __name__ == "__main__":
    main()
