"""Agent that asks an LLM to decompose work into structured executable steps."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from context_analyzer.agents.base import BaseAgent
from context_analyzer.config.settings import AppSettings
from context_analyzer.models.schemas import DecompositionResult


class DecompositionAgent(BaseAgent):
    """Compose prompt, call model, and enrich each step with embeddings."""

    def __init__(self, settings: AppSettings) -> None:
        """Initialize model clients for chat and embeddings."""

        self._llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
        self._embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
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
        result: DecompositionResult = chain.invoke(
            {
                "task_request": state["task_request"],
                "jira_context": state["jira_context"],
            }
        )

        for step in result.steps:
            vector = self._embeddings.embed_query(step.step_description)
            step.embedding = vector

        return {"decomposition": result.model_dump()}
