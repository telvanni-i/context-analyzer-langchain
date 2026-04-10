"""Factory for the LangGraph workflow used by this project."""

from __future__ import annotations

from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from context_analyzer.graph.nodes import (
    decompose_task_node,
    read_jira_context_node,
    read_request_node,
)
from context_analyzer.graph.state import WorkflowState


def build_workflow() -> CompiledStateGraph[WorkflowState, None, WorkflowState, WorkflowState]:
    """Build and compile the three-step graph.

    Flow:
        1) Read request file.
        2) Read JIRA context file.
        3) Ask the LLM to create structured algorithm steps + tags + embeddings.
    """

    graph = StateGraph(WorkflowState)
    graph.add_node("read_request", cast(Any, read_request_node))
    graph.add_node("read_jira_context", cast(Any, read_jira_context_node))
    graph.add_node("decompose", cast(Any, decompose_task_node))

    graph.add_edge(START, "read_request")
    graph.add_edge("read_request", "read_jira_context")
    graph.add_edge("read_jira_context", "decompose")
    graph.add_edge("decompose", END)

    return graph.compile()
