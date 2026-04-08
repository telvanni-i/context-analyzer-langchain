"""Application settings loaded from `.env.local` and environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppSettings:
    """Runtime configuration used by the graph.

    Attributes:
        openai_api_key: Secret key for OpenAI API calls.
        openai_model: Model name used for task decomposition.
        embedding_model: Embedding model used for step vector generation.
    """

    openai_api_key: str
    openai_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"


def load_settings() -> AppSettings:
    """Load settings from `.env.local` with safe defaults.

    Returns:
        Parsed immutable settings object.

    Raises:
        ValueError: If required secrets are missing.
    """

    load_dotenv(".env.local")

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY in .env.local or environment.")

    return AppSettings(
        openai_api_key=api_key,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip(),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip(),
    )
