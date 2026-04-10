"""CLI entry point for running the context analyzer graph."""

from __future__ import annotations

import io

from PIL import Image

from context_analyzer.graph.workflow import build_workflow


def main() -> None:
    """Execute graph and persist decomposition result."""

    img = build_workflow().get_graph().draw_png()  # type: ignore[call-overload]

    Image.open(io.BytesIO(img)).show()


if __name__ == "__main__":
    main()
