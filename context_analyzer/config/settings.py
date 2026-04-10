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
        socks5_url: Optional SOCKS5 proxy URL used for OpenAI HTTP calls.
        openai_logs_path: Local file path where every LLM interaction is appended.
        use_mock_openai: Toggle for mock agent implementation without network calls.
    """

    openai_api_key: str
    openai_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    socks5_url: str | None = None
    openai_logs_path: str = ""
    use_mock_openai: bool = False


def load_settings() -> AppSettings:
    """Load settings from `.env.local` with safe defaults.

    Returns:
        Parsed immutable settings object.

    Raises:
        ValueError: If required secrets or config are missing.
    """

    load_dotenv(".env.local")

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY in .env.local or environment.")

    logs_path = os.getenv("OPENAI_LOGS_PATH", "").strip()
    if not logs_path:
        raise ValueError("Missing OPENAI_LOGS_PATH in .env.local or environment.")

    socks5_url = os.getenv("SOCKS5_URL", "").strip() or None

    use_mock_openai = os.getenv("USE_MOCK_OPENAI", "").strip().lower() in {"1", "true", "yes", "on"}

    return AppSettings(
        openai_api_key=api_key,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip(),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip(),
        socks5_url=socks5_url,
        openai_logs_path=logs_path,
        use_mock_openai=use_mock_openai,
    )
