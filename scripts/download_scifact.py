"""
Script to acquire and verify the SciFact dataset from its authoritative public source.

Dataset Details:
- Dataset: SciFact (Scientific Claim Verification)
- Paper: "Fact or Fiction: Verifying Scientific Claims" (EMNLP 2020)
  Wadden et al., Allen Institute for AI (AI2) & University of Washington
- Source URL: https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz
- Code & Dataset Repo: https://github.com/allenai/scifact
- License: Apache License 2.0 (code/dataset release by AllenAI)
- Artifacts:
  * corpus.jsonl: Research paper abstracts with doc_id, title, abstract sentences
  * claims_train.jsonl: Training claims with doc-level and sentence-level evidence annotations
  * claims_dev.jsonl: Validation claims with gold evidence annotations
  * claims_test.jsonl: Test claims (note: official labels held out on leaderboard)
"""

import os
import tarfile
import urllib.request
import json
from pathlib import Path

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"

def download_and_extract_scifact(target_dir: str = "data/raw") -> dict:
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    tar_path = target_path / "data.tar.gz"

    expected_files = ["corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl", "claims_test.jsonl"]
    all_exist = all((target_path / fname).exists() for fname in expected_files)

    if not all_exist:
        print(f"Downloading SciFact archive from {SCIFACT_URL}...")
        urllib.request.urlretrieve(SCIFACT_URL, tar_path)
        print(f"Downloaded archive to {tar_path} ({tar_path.stat().st_size / (1024 * 1024):.2f} MB).")

        print("Extracting archive...")
        with tarfile.open(tar_path, "r:gz") as tar:
            # Files in tar are prefixed with 'data/'
            for member in tar.getmembers():
                filename = os.path.basename(member.name)
                if filename in expected_files:
                    member_file = tar.extractfile(member)
                    if member_file:
                        dest_file = target_path / filename
                        with open(dest_file, "wb") as f_out:
                            f_out.write(member_file.read())
                        print(f"Extracted: {dest_file.name}")
    else:
        print(f"SciFact data files already present in {target_path.resolve()}. Skipping download.")

    # Validation and statistics summary
    stats = {}
    for fname in expected_files:
        fpath = target_path / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing expected dataset file: {fpath}")

        count = 0
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        stats[fname] = count
        print(f"  - {fname}: {count:,} records ({fpath.stat().st_size / 1024:.1f} KB)")

    return stats

if __name__ == "__main__":
    stats = download_and_extract_scifact()
    print("SciFact acquisition complete. Summary:")
    print(json.dumps(stats, indent=2))
