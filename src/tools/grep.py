"""Read-only regular-expression search tool for CodePin repositories."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
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


MAX_RESULTS = 100
SEARCH_TIMEOUT_SECONDS = 30
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024


def _require_ripgrep() -> str:
    executable = shutil.which("rg")
    if executable is None:
        raise RuntimeError(
            "ripgrep (rg) is required for CodePin grep; install it before starting"
        )
    return executable


class GrepAction(Action):
    """Minimal schema for searching repository file contents."""

    pattern: str = Field(min_length=1, description="Regular expression to search for")
    path: str | None = Field(
        default=None,
        description=(
            "Optional repository-relative file or directory. "
            "Defaults to the repository root."
        ),
    )
    include: str | None = Field(
        default=None,
        description='Optional glob filter such as "*.py" or "*.{ts,tsx}".',
    )


class GrepObservation(Observation):
    """Structured grep result with path, line number, and source line."""

    matches: list[str] = Field(description="Formatted matching source lines")
    pattern: str = Field(description="Regular expression used for the search")
    search_path: str = Field(description="Repository-relative path searched")
    include_pattern: str | None = Field(description="Optional file glob filter")
    truncated: bool = Field(description="Whether more than 100 lines matched")


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


def _relative_path(workspace_root: Path, value: str) -> str | None:
    try:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = workspace_root / candidate
        return candidate.resolve().relative_to(workspace_root).as_posix()
    except (OSError, ValueError):
        return None


class GrepExecutor(ToolExecutor[GrepAction, GrepObservation]):
    """Run deterministic, bounded searches through the required ripgrep backend."""

    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root).resolve()
        self.rg_executable = _require_ripgrep()

    def __call__(
        self,
        action: GrepAction,
        conversation: LocalConversation | None = None,
    ) -> GrepObservation:
        try:
            search_path = _resolve_search_path(self.workspace_root, action.path)
            if not search_path.exists():
                raise ValueError(f"search path does not exist: {action.path or '.'}")

            matches, truncated = self._search_with_ripgrep(action, search_path)

            relative_search_path = search_path.relative_to(
                self.workspace_root
            ).as_posix()
            relative_search_path = relative_search_path or "."
            text = "\n".join(matches) if matches else "No matches found."
            if truncated:
                text += (
                    f"\n\n[Results truncated to {MAX_RESULTS} matching lines; "
                    "use a narrower pattern, path, or include filter.]"
                )
            return GrepObservation.from_text(
                text=text,
                matches=matches,
                pattern=action.pattern,
                search_path=relative_search_path,
                include_pattern=action.include,
                truncated=truncated,
            )
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            return GrepObservation.from_text(
                text=str(exc),
                is_error=True,
                matches=[],
                pattern=action.pattern,
                search_path=action.path or ".",
                include_pattern=action.include,
                truncated=False,
            )

    def _search_with_ripgrep(
        self,
        action: GrepAction,
        search_path: Path,
    ) -> tuple[list[str], bool]:
        command = [
            self.rg_executable,
            "--json",
            "--color=never",
            "--sort=path",
            f"--max-filesize={MAX_FILE_SIZE_BYTES}",
            "--regexp",
            action.pattern,
        ]
        if action.include:
            command.extend(["--glob", action.include])
        relative_search_path = search_path.relative_to(self.workspace_root).as_posix()
        command.append(relative_search_path or ".")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=self.workspace_root,
        )

        def collect_matches() -> tuple[list[str], bool]:
            assert process.stdout is not None
            matches: list[str] = []
            for raw_line in process.stdout:
                try:
                    event = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") != "match":
                    continue
                data = event.get("data", {})
                path_text = data.get("path", {}).get("text")
                line_number = data.get("line_number")
                line_text = data.get("lines", {}).get("text")
                if not path_text or line_number is None or line_text is None:
                    continue
                relative = _relative_path(self.workspace_root, path_text)
                if relative is None:
                    continue
                source_line = line_text.rstrip("\r\n")
                matches.append(f"{relative}:{line_number}:{source_line}")
                if len(matches) > MAX_RESULTS:
                    if process.poll() is None:
                        process.terminate()
                    return matches[:MAX_RESULTS], True
            return matches, False

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(collect_matches)
                try:
                    matches, truncated = future.result(timeout=SEARCH_TIMEOUT_SECONDS)
                except FutureTimeoutError as exc:
                    process.kill()
                    raise subprocess.TimeoutExpired(
                        command,
                        SEARCH_TIMEOUT_SECONDS,
                    ) from exc
            return_code = process.wait(timeout=5)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

        assert process.stderr is not None
        stderr = process.stderr.read().strip()
        if not truncated and return_code not in {0, 1}:
            detail = stderr or f"ripgrep exited with {return_code}"
            raise ValueError(detail)
        return matches, truncated


TOOL_DESCRIPTION = """Search repository file contents with a regular expression.

The result contains repository-relative file paths, 1-based line numbers, and the
matching source lines. Use the optional path to narrow the search and include to
filter file names. Results are limited to 100 matching lines. Requires ripgrep (`rg`).
"""


class GrepTool(ToolDefinition[GrepAction, GrepObservation]):
    """OpenHands definition for CodePin's content search."""

    @classmethod
    def create(
        cls,
        conv_state: ConversationState,
        **params,
    ) -> Sequence[GrepTool]:
        if params:
            raise ValueError("GrepTool does not accept initialization parameters")
        working_dir = conv_state.workspace.working_dir
        if not Path(working_dir).is_dir():
            raise ValueError(f"working_dir {working_dir!r} is not a valid directory")
        return [
            cls(
                description=TOOL_DESCRIPTION,
                action_type=GrepAction,
                observation_type=GrepObservation,
                executor=GrepExecutor(working_dir),
                annotations=ToolAnnotations(
                    title="grep",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            )
        ]


register_tool(GrepTool.name, GrepTool)
