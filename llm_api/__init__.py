from typing import Any

from .base import LLMAPI, LLMMsg, LLMMsgRole, LLMMsgStats

__all__ = [
    'LLMAPI', 'LLMMsg', 'LLMMsgRole', 'LLMMsgStats', 'MockLLMAPI', 'OpenAIAPI'
]


def __getattr__(name: str) -> Any:
    if name == 'OpenAIAPI':
        from .openai_api import OpenAIAPI

        return OpenAIAPI
    if name == 'MockLLMAPI':
        from .mock import MockLLMAPI

        return MockLLMAPI
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
