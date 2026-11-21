import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Base directory is the ResearchMap root (parent of backend)
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env if present
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

class Config:
    """Configurable project settings using relative paths and environment overrides."""
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / os.getenv("DATA_DIR", "data")
    RAW_DATA_DIR: Path = BASE_DIR / os.getenv("RAW_DATA_DIR", "data/raw")
    PROCESSED_DATA_DIR: Path = BASE_DIR / os.getenv("PROCESSED_DATA_DIR", "data/processed")
    MODELS_DIR: Path = BASE_DIR / os.getenv("MODELS_DIR", "models")
    REPORTS_DIR: Path = BASE_DIR / os.getenv("REPORTS_DIR", "reports")

    # Retrieval and NLI settings (Configuration C)
    BM25_TOP_K_DOCS: int = int(os.getenv("BM25_TOP_K_DOCS", "10"))
    BM25_TOP_K_SENTENCES: int = int(os.getenv("BM25_TOP_K_SENTENCES", "4"))
    BM25_DOC_SCORE_WEIGHT: float = float(os.getenv("BM25_DOC_SCORE_WEIGHT", "0.35"))
    BM25_MAX_SENTENCES_PER_DOC: Optional[int] = (
        int(os.getenv("BM25_MAX_SENTENCES_PER_DOC", "2"))
        if os.getenv("BM25_MAX_SENTENCES_PER_DOC", "2").lower() not in ("none", "0")
        else None
    )
    NLI_MODEL_NAME: str = os.getenv("NLI_MODEL_NAME", "cross-encoder/nli-deberta-v3-small")
    DEVICE: str = os.getenv("DEVICE", "auto")
    RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))

    # Entailment decision thresholds
    # When cross-encoder outputs probabilities:
    # If P(SUPPORT) > ENTAIL_THRESHOLD -> SUPPORTS
    # If P(CONTRADICT) > CONTRADICT_THRESHOLD -> CONTRADICTS
    # Otherwise -> NOT ENOUGH INFORMATION
    ENTAIL_THRESHOLD: float = float(os.getenv("ENTAIL_THRESHOLD", "0.50"))
    CONTRADICT_THRESHOLD: float = float(os.getenv("CONTRADICT_THRESHOLD", "0.50"))

    @classmethod
    def ensure_directories(cls):
        """Ensure all required output and cache directories exist."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
