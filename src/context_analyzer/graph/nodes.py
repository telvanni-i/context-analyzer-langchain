"""Graph node functions that keep orchestration easy to follow."""

from __future__ import annotations

from context_analyzer.agents.decomposition_agent import DecompositionAgent
from context_analyzer.config.settings import load_settings
from context_analyzer.tools.file_reader import FileReaderTool


def read_request_node(state: dict) -> dict:
    """Read user request from the path provided in graph input state."""

    reader = FileReaderTool()
    return {"task_request": reader.run(state["request_path"])}


def read_jira_context_node(_: dict) -> dict:
    """Read project context from the fixed JIRA context file."""

    reader = FileReaderTool()
    return {"jira_context": reader.run("context/JIRA.txt")}


def decompose_task_node(state: dict) -> dict:
    """Use LLM + embeddings to transform text into step-by-step algorithm output."""

    agent = DecompositionAgent(settings=load_settings())
    return agent.run(state)
