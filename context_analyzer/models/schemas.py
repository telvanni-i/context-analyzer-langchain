"""Data models used by the graph nodes and agent outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StepItem(BaseModel):
    """Single algorithm step enriched for later retrieval and filtering.

    Attributes:
        step_description: Human-readable instruction describing the action.
        tags: Classification labels that help route the step to specialized workers.
        embedding: Numeric vector representing semantic meaning of the step.
    """

    step_description: str = Field(..., description="Plain-language step instruction.")
    tags: list[str] = Field(default_factory=list, description="Topic labels for this step.")
    embedding: list[float] = Field(
        default_factory=list,
        description="Embedding vector for semantic search and clustering.",
    )


class DecompositionResult(BaseModel):
    """Structured output returned by the decomposition agent.

    Attributes:
        steps: Ordered list of decomposed task steps.
    """

    steps: list[StepItem] = Field(default_factory=list)
