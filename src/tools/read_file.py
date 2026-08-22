"""Read-only, line-oriented file reader for CodePin repositories."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from pydantic import Field

if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
    from openhands.sdk.conversation.state import ConversationState


DEFAULT_LINES = 200
MAX_LINES = 500
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024
MAX_OUTPUT_CHARS = 30_000


class ReadFileAction(Action):
    """Schema for reading a repository file or a selected line range."""

    path: str = Field(min_length=1, description="Repository-relative file path")
    start_line: int | None = Field(
        default=None,
        ge=1,
        description="Optional 1-based starting line",
    )
    end_line: int | None = Field(
        default=None,
        ge=1,
        description="Optional 1-based inclusive ending line",
    )


class ReadFileObservation(Observation):
    """Structured result for a line-oriented file read."""

    path: str = Field(description="Repository-relative file path")
    start_line: int = Field(description="First returned line, or 1 for an empty file")
    end_line: int = Field(description="Last returned line, or 0 for an empty file")
    total_lines: int = Field(description="Total number of lines in the file")
    truncated: bool = Field(description="Whether the requested content was truncated")


def _resolve_file(workspace_root: Path, value: str) -> Path:
    requested = Path(value)
    if requested.is_absolute():
        raise ValueError("path must be relative to the repository root")
    if ".." in PurePosixPath(value.replace("\\", "/")).parts:
        raise ValueError("path cannot contain '..'")
    resolved = (workspace_root / requested).resolve()
    try:
        resolved.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError("path must stay within the repository root") from exc
    return resolved


class ReadFileExecutor(ToolExecutor[ReadFileAction, ReadFileObservation]):
    """Read bounded source ranges while preserving accurate line metadata."""

    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root).resolve()

    def __call__(
        self,
        action: ReadFileAction,
        conversation: LocalConversation | None = None,
    ) -> ReadFileObservation:
        start_line = action.start_line or 1
        end_line = action.end_line or (start_line + DEFAULT_LINES - 1)
        try:
            if end_line < start_line:
                raise ValueError("end_line must be greater than or equal to start_line")
            if end_line - start_line + 1 > MAX_LINES:
                raise ValueError(
                    f"a single read_file call can return at most {MAX_LINES} lines"
                )

            file_path = _resolve_file(self.workspace_root, action.path)
            if not file_path.exists():
                raise ValueError(f"file does not exist: {action.path}")
            if not file_path.is_file():
                raise ValueError(f"path is not a file: {action.path}")
            if file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
                raise ValueError(
                    f"file exceeds the {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MiB read limit"
                )

            with file_path.open("rb") as binary_handle:
                if b"\x00" in binary_handle.read(8192):
                    raise ValueError("binary files are not supported")

            selected: list[tuple[int, str]] = []
            total_lines = 0
            output_chars = 0
            output_truncated = False
            with file_path.open("r", encoding="utf-8", errors="replace") as handle:
                for line_number, line in enumerate(handle, start=1):
                    total_lines = line_number
                    if line_number < start_line or line_number > end_line:
                        continue
                    source_line = line.rstrip("\r\n")
                    remaining = MAX_OUTPUT_CHARS - output_chars
                    if remaining <= 0:
                        output_truncated = True
                        continue
                    if len(source_line) > remaining:
                        source_line = source_line[:remaining]
                        output_truncated = True
                    selected.append((line_number, source_line))
                    output_chars += len(source_line)

            if total_lines and start_line > total_lines:
                raise ValueError(
                    f"start_line {start_line} exceeds the file's {total_lines} lines"
                )

            relative = Path(action.path).as_posix()
            actual_start = selected[0][0] if selected else 1
            actual_end = selected[-1][0] if selected else 0
            range_truncated = bool(
                action.end_line is None and total_lines and end_line < total_lines
            )
            truncated = output_truncated or range_truncated

            if selected:
                body = "\n".join(f"{number:6d}→{line}" for number, line in selected)
                text = (
                    f"File: {relative} (lines {actual_start}-{actual_end} "
                    f"of {total_lines})\n{body}"
                )
            else:
                text = f"File: {relative} is empty."
            if truncated:
                if output_truncated:
                    text += (
                        "\n\n[Content truncated by the output character limit; "
                        "request a narrower line range.]"
                    )
                else:
                    next_line = actual_end + 1
                    text += (
                        "\n\n[Content truncated by the default line limit. "
                        f"Use start_line={next_line} to continue reading.]"
                    )

            return ReadFileObservation.from_text(
                text=text,
                path=relative,
                start_line=actual_start,
                end_line=actual_end,
                total_lines=total_lines,
                truncated=truncated,
            )
        except (OSError, ValueError) as exc:
            return ReadFileObservation.from_text(
                text=str(exc),
                is_error=True,
                path=action.path,
                start_line=start_line,
                end_line=0,
                total_lines=0,
                truncated=False,
            )


TOOL_DESCRIPTION = """Read a text file from the repository with line numbers.

The path must be relative to the repository root. Use start_line and end_line for
targeted reads; both are 1-based and end_line is inclusive. The default read is 200
lines and one call can return at most 500 lines.
"""


class ReadFileTool(ToolDefinition[ReadFileAction, ReadFileObservation]):
    """OpenHands definition for CodePin's bounded file reader."""

    @classmethod
    def create(
        cls,
        conv_state: ConversationState,
        **params,
    ) -> Sequence[ReadFileTool]:
        if params:
            raise ValueError("ReadFileTool does not accept initialization parameters")
        working_dir = conv_state.workspace.working_dir
        if not Path(working_dir).is_dir():
            raise ValueError(f"working_dir {working_dir!r} is not a valid directory")
        return [
            cls(
                description=TOOL_DESCRIPTION,
                action_type=ReadFileAction,
                observation_type=ReadFileObservation,
                executor=ReadFileExecutor(working_dir),
                annotations=ToolAnnotations(
                    title="read_file",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            )
        ]


register_tool(ReadFileTool.name, ReadFileTool)
