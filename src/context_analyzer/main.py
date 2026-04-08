"""CLI entry point for running the context analyzer graph."""

from __future__ import annotations

import argparse

from context_analyzer.graph.workflow import build_workflow
from context_analyzer.utils.io import write_json


def parse_args() -> argparse.Namespace:
    """Parse command-line flags for input request and output files."""

    parser = argparse.ArgumentParser(description="Run task decomposition workflow.")
    parser.add_argument(
        "--request-file",
        default="input/user_request.txt",
        help="Path to a text file containing the user request.",
    )
    parser.add_argument(
        "--output-file",
        default="output/decomposition.json",
        help="Path where the structured decomposition JSON is written.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute graph and persist decomposition result."""

    args = parse_args()
    workflow = build_workflow()
    result = workflow.invoke({"request_path": args.request_file})
    write_json(args.output_file, result["decomposition"])
    print(f"Saved decomposition to {args.output_file}")


if __name__ == "__main__":
    main()
