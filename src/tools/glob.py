"""Read-only glob tool tailored to CodePin localization workspaces."""

from __future__ import annotations

from bisect import insort
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

from src.profiling import nvtx_range

if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
    from openhands.sdk.conversation.state import ConversationState


MAX_RESULTS = 100
MAX_CANDIDATES = MAX_RESULTS + 1
IGNORED_PARTS = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "node_modules",
    }
)


class GlobAction(Action):
    """Schema for finding files by path pattern."""

    pattern: str = Field(
        min_length=1,
        description='Glob pattern relative to the search path, for example "**/*.py".',
    )
    path: str | None = Field(
        default=None,
        description=(
            "Optional repository-relative directory to search. "
            "Defaults to the repository root."
        ),
    )


class GlobObservation(Observation):
    """Structured glob result."""

    files: list[str] = Field(description="Matching repository-relative file paths")
    pattern: str = Field(description="Glob pattern used for the search")
    search_path: str = Field(description="Repository-relative directory searched")
    truncated: bool = Field(description="Whether more than 100 files matched")


def _resolve_search_path(workspace_root: Path, value: str | None) -> Path:
    if value is None or value in {"", "."}:
        return workspace_root

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


def _validate_pattern(pattern: str) -> None:
    normalized = pattern.replace("\\", "/")
    parsed = PurePosixPath(normalized)
    if parsed.is_absolute():
        raise ValueError("pattern must be relative to the search path")
    if ".." in parsed.parts:
        raise ValueError("pattern cannot contain '..'")


class GlobExecutor(ToolExecutor[GlobAction, GlobObservation]):
    """Find files without allowing reads outside the cloned repository."""

    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root).resolve()

    @nvtx_range("codepin.tool.glob")
    def __call__(
        self,
        action: GlobAction,
        conversation: LocalConversation | None = None,
    ) -> GlobObservation:
        try:
            _validate_pattern(action.pattern)
            search_path = _resolve_search_path(self.workspace_root, action.path)
            if not search_path.is_dir():
                raise ValueError(
                    f"search path is not a directory: {action.path or '.'}"
                )

            candidates: list[str] = []
            for candidate in search_path.glob(action.pattern):
                try:
                    resolved = candidate.resolve()
                    resolved.relative_to(self.workspace_root)
                    relative = candidate.relative_to(self.workspace_root)
                except (OSError, ValueError):
                    continue
                if not resolved.is_file() or IGNORED_PARTS.intersection(relative.parts):
                    continue
                relative_path = relative.as_posix()
                if relative_path in candidates:
                    continue
                insort(candidates, relative_path)
                if len(candidates) > MAX_CANDIDATES:
                    candidates.pop()

            truncated = len(candidates) > MAX_RESULTS
            files = candidates[:MAX_RESULTS]
            relative_search_path = search_path.relative_to(
                self.workspace_root
            ).as_posix()
            relative_search_path = relative_search_path or "."

            if files:
                text = "\n".join(files)
                if truncated:
                    text += (
                        f"\n\n[Results truncated to {MAX_RESULTS} files; "
                        "use a narrower pattern or path.]"
                    )
            else:
                text = (
                    f"No files found matching {action.pattern!r} "
                    f"under {relative_search_path!r}."
                )

            return GlobObservation.from_text(
                text=text,
                files=files,
                pattern=action.pattern,
                search_path=relative_search_path,
                truncated=truncated,
            )
        except (OSError, ValueError) as exc:
            return GlobObservation.from_text(
                text=str(exc),
                is_error=True,
                files=[],
                pattern=action.pattern,
                search_path=action.path or ".",
                truncated=False,
            )


TOOL_DESCRIPTION = """Find files in the repository using a glob pattern.

Use this tool to discover candidate files by name or extension. Paths in both the
arguments and results are relative to the repository root. Results are sorted for
deterministic behavior and limited to 100 files.
"""


class GlobTool(ToolDefinition[GlobAction, GlobObservation]):
    """OpenHands definition for CodePin's glob search."""

    @classmethod
    def create(
        cls,
        conv_state: ConversationState,
        **params,
    ) -> Sequence[GlobTool]:
        if params:
            raise ValueError("GlobTool does not accept initialization parameters")
        working_dir = conv_state.workspace.working_dir
        if not Path(working_dir).is_dir():
            raise ValueError(f"working_dir {working_dir!r} is not a valid directory")
        return [
            cls(
                description=TOOL_DESCRIPTION,
                action_type=GlobAction,
                observation_type=GlobObservation,
                executor=GlobExecutor(working_dir),
                annotations=ToolAnnotations(
                    title="glob",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            )
        ]


register_tool(GlobTool.name, GlobTool)
