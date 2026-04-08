"""Abstract base types for tools that interact with files or services."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTool(ABC):
    """Base class for simple reusable tool objects."""

    @abstractmethod
    def run(self, *args, **kwargs):
        """Execute tool logic and return a value."""
