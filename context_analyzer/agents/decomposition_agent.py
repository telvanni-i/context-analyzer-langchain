"""Agent that asks an LLM to decompose work into structured executable steps."""

from __future__ import annotations

import hashlib

import httpx
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from context_analyzer.agents.base import BaseAgent
from context_analyzer.config.settings import AppSettings
from context_analyzer.models.schemas import DecompositionResult, StepItem
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

        result = self._generate_decomposition(
            task_request=state["task_request"],
            jira_context=state["jira_context"],
        )

        for step in result.steps:
            step.embedding = self._embed_step(step.step_description)

        return {"decomposition": result.model_dump()}

    def _generate_decomposition(
        self,
        *,
        task_request: str,
        jira_context: str,
    ) -> DecompositionResult:
        """Perform the OpenAI structured-output call for decomposition."""

        chain = self._prompt | self._llm.with_structured_output(DecompositionResult)
        raw_result = chain.invoke(
            {
                "task_request": task_request,
                "jira_context": jira_context,
            }
        )
        return DecompositionResult.model_validate(raw_result)

    def _embed_step(self, step_description: str) -> list[float]:
        """Perform the OpenAI embedding call for a step description."""

        return self._embeddings.embed_query(step_description)


class MockDecompositionAgent(DecompositionAgent):
    """Drop-in replacement that avoids real OpenAI calls for local testing."""

    def _generate_decomposition(
        self,
        *,
        task_request: str,
        jira_context: str,
    ) -> DecompositionResult:
        """Return deterministic mock decomposition steps."""

        return DecompositionResult(
            steps=[
                StepItem(
                    step_description=(
                        "Summarize the request and context into explicit acceptance criteria."
                    ),
                    tags=["analysis", "requirements"],
                ),
                StepItem(
                    step_description=(
                        "Design implementation tasks with dependencies and clear ordering."
                    ),
                    tags=["planning", "architecture"],
                ),
                StepItem(
                    step_description=(
                        "Validate outcomes with tests and produce a concise execution report."
                    ),
                    tags=["testing", "reporting"],
                ),
            ]
        )

    def _embed_step(self, step_description: str) -> list[float]:
        """Return a deterministic pseudo-embedding derived from step text."""

        digest = hashlib.sha256(step_description.encode("utf-8")).digest()
        return [round(byte / 255.0, 6) for byte in digest[:8]]
