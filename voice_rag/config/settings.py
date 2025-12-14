from dataclasses import dataclass
from pathlib import Path

# Project root = folder where app_streamlit.py lives
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "voice_rag" / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = DATA_DIR / "chroma_store"

# Ensure folders exist at import time
for p in (DATA_DIR, PDF_DIR, CHROMA_DIR):
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    """Central place for model/config knobs used by the app."""

    ollama_model: str = "llama3"  # adjust if you use a different model
    chroma_dir: Path = CHROMA_DIR
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


settings = Settings()
