from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path

import httpx
from openai import AsyncOpenAI

from protocol.eae_json_schema import EVENTS_AND_ENTITIES_JSON_SCHEMA

from .base import LLMAPI, LLMMsg, LLMMsgRole, LLMMsgStats

_OPENAI_EAE_RESPONSE_FORMAT = {
    'type': 'json_schema',
    'name': 'events_and_entities',
    'schema': EVENTS_AND_ENTITIES_JSON_SCHEMA,
    'strict': False,
}


class OpenAIAPI(LLMAPI):
    def __init__(
        self,
        log_dir: str | Path,
        api_key: str | None = None,
        flex_mode: bool = False,
        model: str = 'gpt-5',
        proxy_uri: str | None = None
    ):
        super().__init__(log_dir)
        self.flex_mode = flex_mode
        self.model = model

        if api_key is None:
            api_key = os.environ.get('OPENAI_API_KEY')
        if proxy_uri is None:
            proxy_uri = os.environ.get('LLM_PROXY_URI')

        custom_client = httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(proxy=proxy_uri) if proxy_uri else None)

        self._client = AsyncOpenAI(
            api_key=api_key,
            http_client=custom_client
        )

    async def _request(self, messages: list[LLMMsg]) -> tuple[LLMMsg, LLMMsgStats]:
        response = await self._client.responses.create(
            model=self.model,
            service_tier='flex',
            input=[{'role': message.role, 'content': message.content} for message in messages],
            text={'format': _OPENAI_EAE_RESPONSE_FORMAT},
        )
        text = response.output_text
        usage = response.usage
        input_tokens = self._read_usage_value(usage.input_tokens)
        cached_tokens = self._read_usage_value(getattr(usage, 'input_tokens_details', None), 'cached_tokens')
        output_tokens = self._read_usage_value(usage.output_tokens)
        return (
            LLMMsg(role=LLMMsgRole.assistant, content=text),
            LLMMsgStats(
                input_tokens=input_tokens,
                input_cached_tokens=cached_tokens,
                output_tokens=output_tokens,
                price_usd=Decimal('0'),
            ),
        )

    @staticmethod
    def _read_usage_value(value: object, field_name: str | None = None) -> int:
        if value is None:
            return 0
        if field_name is not None:
            nested_value = getattr(value, field_name, None)
            return int(nested_value) if isinstance(nested_value, int | str) else 0
        return int(value) if isinstance(value, int | str) else 0
