"""Utilities for writing detailed LLM request/response logs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from langchain_core.callbacks.base import BaseCallbackHandler


class OpenAILogCallbackHandler(BaseCallbackHandler):
    """Append each chat interaction to a JSON-lines log file."""

    def __init__(self, log_path: str) -> None:
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[Any]], **kwargs: Any) -> None:
        """Write prompt messages when chat generation starts."""

        self._append(
            {
                "timestamp": self._timestamp(),
                "event": "chat_model_start",
                "serialized": serialized,
                "messages": [
                    [
                        {
                            "type": getattr(message, "type", None),
                            "content": getattr(message, "content", None),
                            "additional_kwargs": getattr(message, "additional_kwargs", {}),
                        }
                        for message in message_list
                    ]
                    for message_list in messages
                ],
                "metadata": kwargs,
            }
        )

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Write model output payload when generation completes."""

        generations: list[list[dict[str, Any]]] = []
        for generation_group in getattr(response, "generations", []):
            row: list[dict[str, Any]] = []
            for generation in generation_group:
                message = getattr(generation, "message", None)
                row.append(
                    {
                        "text": getattr(generation, "text", None),
                        "generation_info": getattr(generation, "generation_info", None),
                        "message": {
                            "type": getattr(message, "type", None),
                            "content": getattr(message, "content", None),
                            "additional_kwargs": getattr(message, "additional_kwargs", None),
                        }
                        if message is not None
                        else None,
                    }
                )
            generations.append(row)

        self._append(
            {
                "timestamp": self._timestamp(),
                "event": "llm_end",
                "generations": generations,
                "llm_output": getattr(response, "llm_output", None),
                "run": kwargs,
            }
        )

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Persist LLM errors to the same log stream."""

        self._append(
            {
                "timestamp": self._timestamp(),
                "event": "llm_error",
                "error_type": type(error).__name__,
                "error": str(error),
                "run": kwargs,
            }
        )

    def _append(self, entry: dict[str, Any]) -> None:
        with self._lock:
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False, default=str))
                handle.write("\n")

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()
