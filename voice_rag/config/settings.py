from dataclasses import dataclass
from pathlib import Path

# Project root = folder where app_streamlit.py will live
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "voice_rag" / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = DATA_DIR / "chroma_store"

# Ensure folders exist
for p in (DATA_DIR, PDF_DIR, CHROMA_DIR):
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    ollama_model: str = "llama3"   # adjust if you use a different model
    chroma_dir: Path = CHROMA_DIR
    # we’ll start with this multilingual embedding model
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


settings = Settings()
