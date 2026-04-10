from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import aiofiles
from pydantic import BaseModel, ConfigDict


class LLMMsgRole:
    system = 'system'
    assistant = 'assistant'
    user = 'user'


class LLMMsg(BaseModel):
    role: str
    content: str
    tokens: int | None = None

    model_config = ConfigDict(extra='forbid')


class LLMMsgStats(BaseModel):
    input_tokens: int
    input_cached_tokens: int
    output_tokens: int
    price_usd: Decimal

    model_config = ConfigDict(extra='forbid')


class LLMAPI(ABC):
    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)

    async def request(self, messages: list[LLMMsg]) -> LLMMsg:
        response, stats = await self._request(messages)
        await self._store_interaction(messages, response, stats)
        return response

    async def _store_interaction(
        self, messages: list[LLMMsg], response: LLMMsg, stats: LLMMsgStats
    ) -> None:
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        interaction_dir = self.log_dir / timestamp
        interaction_dir.mkdir(parents=True, exist_ok=False)

        await self._write_json(interaction_dir / 'request.json', [message.model_dump(exclude_none=True) for message in messages])
        await self._write_json(interaction_dir / 'response.json', response.model_dump(exclude_none=True))
        await self._write_json(interaction_dir / 'stats.json', stats.model_dump(mode='json'))

    async def _write_json(self, path: Path, payload: object) -> None:
        async with aiofiles.open(path, 'w', encoding='utf-8') as file:
            await file.write(json.dumps(payload, ensure_ascii=False, indent=2))
            await file.write('\n')

    @abstractmethod
    async def _request(self, messages: list[LLMMsg]) -> tuple[LLMMsg, LLMMsgStats]:
        raise NotImplementedError
