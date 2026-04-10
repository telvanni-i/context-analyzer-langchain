"""Abstract base types for agents used in the workflow graph."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Common interface for all agents.

    Concrete agents should implement a single `run` method. This keeps each file
    easy to read and enables swapping implementations without changing graph code.
    """

    @abstractmethod
    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute agent logic and return state updates.

        Args:
            state: Current graph state dictionary.

        Returns:
            Partial state update merged by the graph runtime.
        """
