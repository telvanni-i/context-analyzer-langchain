"""Graph node functions that keep orchestration easy to follow."""

from __future__ import annotations

from context_analyzer.agents.decomposition_agent import (
    DecompositionAgent,
    MockDecompositionAgent,
)
from context_analyzer.config.settings import load_settings
from context_analyzer.graph.state import WorkflowState
from context_analyzer.tools.file_reader import FileReaderTool


def read_request_node(state: WorkflowState) -> WorkflowState:
    """Read user request from the path provided in graph input state."""

    reader = FileReaderTool()
    return {"task_request": reader.run(state["request_path"])}


def read_jira_context_node(_: WorkflowState) -> WorkflowState:
    """Read project context from the fixed JIRA context file."""

    reader = FileReaderTool()
    return {"jira_context": reader.run("context/JIRA.txt")}


def decompose_task_node(state: WorkflowState) -> WorkflowState:
    """Use LLM + embeddings to transform text into step-by-step algorithm output."""

    settings = load_settings()
    agent_cls = MockDecompositionAgent if settings.use_mock_openai else DecompositionAgent
    agent = agent_cls(settings=settings)
    return agent.run(state)
