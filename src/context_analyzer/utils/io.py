"""I/O helpers for writing workflow outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: str, payload: dict[str, Any]) -> None:
    """Write JSON output with stable formatting for easy diff/review."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
