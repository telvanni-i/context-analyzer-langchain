"""Typed state definition used by the LangGraph workflow."""

from __future__ import annotations

from typing import TypedDict


class WorkflowState(TypedDict, total=False):
    """State container passed between graph nodes.

    Fields are optional to support incremental updates by each node.
    """

    request_path: str
    task_request: str
    jira_context: str
    decomposition: dict
