"""Central configuration, loaded from environment variables / a .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    # Embeddings & retrieval
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 800  # characters
    chunk_overlap: int = 150  # characters
    top_k: int = 5
    min_similarity: float = 0.25
    max_upload_mb: int = 50

    # LLM
    llm_provider: str = "huggingface"  # huggingface | ollama | openai | extractive
    max_new_tokens: int = 400
    hf_llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"

    # Storage / frontend
    data_dir: Path = BASE_DIR / "data"
    api_url: str = "http://localhost:8000"

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")


def load_settings() -> Settings:
    d = Settings()
    return Settings(
        embedding_model=os.getenv("EMBEDDING_MODEL", d.embedding_model),
        chunk_size=_int("CHUNK_SIZE", d.chunk_size),
        chunk_overlap=_int("CHUNK_OVERLAP", d.chunk_overlap),
        top_k=_int("TOP_K", d.top_k),
        min_similarity=_float("MIN_SIMILARITY", d.min_similarity),
        max_upload_mb=_int("MAX_UPLOAD_MB", d.max_upload_mb),
        llm_provider=os.getenv("LLM_PROVIDER", d.llm_provider).lower(),
        max_new_tokens=_int("MAX_NEW_TOKENS", d.max_new_tokens),
        hf_llm_model=os.getenv("HF_LLM_MODEL", d.hf_llm_model),
        ollama_url=os.getenv("OLLAMA_URL", d.ollama_url),
        ollama_model=os.getenv("OLLAMA_MODEL", d.ollama_model),
        openai_api_key=os.getenv("OPENAI_API_KEY", d.openai_api_key),
        openai_model=os.getenv("OPENAI_MODEL", d.openai_model),
        openai_base_url=os.getenv("OPENAI_BASE_URL", d.openai_base_url),
        data_dir=Path(os.getenv("DATA_DIR", str(d.data_dir))),
        api_url=os.getenv("API_URL", d.api_url),
    )
