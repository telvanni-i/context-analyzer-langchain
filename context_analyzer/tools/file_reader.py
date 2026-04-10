"""File reader tool used by graph nodes."""

from __future__ import annotations

from pathlib import Path

from context_analyzer.tools.base import BaseTool


class FileReaderTool(BaseTool):
    """Read UTF-8 text files from disk.

    This tool intentionally keeps behavior small and predictable:
    - it resolves absolute paths
    - it fails fast if file does not exist
    - it always returns stripped text
    """

    def run(self, *args: object, **kwargs: object) -> str:
        """Read and return the file content.

        Args:
            path: Relative or absolute file path.

        Returns:
            Text content without leading/trailing whitespace.
        """

        raw_path = kwargs.get("path", args[0] if args else None)
        if not isinstance(raw_path, str):
            raise TypeError("path must be provided as a string argument")

        file_path = Path(raw_path).resolve()
        return file_path.read_text(encoding="utf-8").strip()
