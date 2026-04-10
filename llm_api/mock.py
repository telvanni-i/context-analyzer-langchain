from __future__ import annotations

import json
import os
from pathlib import Path

from .base import LLMAPI, LLMMsg, LLMMsgStats


class MockLLMAPI(LLMAPI):
    def __init__(self, log_dir: str | Path, response_path: str | Path | None = None):
        super().__init__(log_dir)
        self.response_path = Path(response_path) if response_path is not None else (
            mock_path if os.path.exists(mock_path := Path(__file__).with_name(
                'mock_static_response_gitignoreme.json'
            )) else Path(__file__).with_name('mock_static_response.json'))

    async def _request(self, messages: list[LLMMsg]) -> tuple[LLMMsg, LLMMsgStats]:
        del messages
        payload = json.loads(self.response_path.read_text(encoding='utf-8'))
        return LLMMsg.model_validate(payload['response']), LLMMsgStats.model_validate(payload['stats'])
