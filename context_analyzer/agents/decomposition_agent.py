"""Agent that asks an LLM to decompose work into structured executable steps."""

from __future__ import annotations

import httpx
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from context_analyzer.agents.base import BaseAgent
from context_analyzer.config.settings import AppSettings
from context_analyzer.models.schemas import DecompositionResult
from context_analyzer.utils.openai_logging import OpenAILogCallbackHandler


class DecompositionAgent(BaseAgent):
    """Compose prompt, call model, and enrich each step with embeddings."""

    def __init__(self, settings: AppSettings) -> None:
        """Initialize model clients for chat and embeddings."""

        http_client = (
            httpx.Client(proxy=settings.socks5_url)
            if settings.socks5_url
            else httpx.Client()
        )
        callback = OpenAILogCallbackHandler(settings.openai_logs_path)
        api_key = SecretStr(settings.openai_api_key)

        self._llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=api_key,
            temperature=0,
            http_client=http_client,
            callbacks=[callback],
        )
        self._embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=api_key,
            http_client=http_client,
        )

        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a senior solution architect. Decompose tasks into "
                    "small, ordered implementation steps. Add concise tags.",
                ),
                (
                    "human",
                    "Task request:\n{task_request}\n\n"
                    "JIRA context:\n{jira_context}\n\n"
                    "Return 3-10 steps in execution order.",
                ),
            ]
        )

    def run(self, state: dict) -> dict:
        """Generate step decomposition and embedding vectors.

        Expected state keys:
            task_request: User request text from input file.
            jira_context: Background text read from context/JIRA.txt.
        """

        chain = self._prompt | self._llm.with_structured_output(DecompositionResult)
        raw_result = chain.invoke(
            {
                "task_request": state["task_request"],
                "jira_context": state["jira_context"],
            }
        )
        result = DecompositionResult.model_validate(raw_result)

        for step in result.steps:
            vector = self._embeddings.embed_query(step.step_description)
            step.embedding = vector

        return {"decomposition": result.model_dump()}
