#!/usr/bin/env python3
"""Generate resumable, gold-validated CodePin SFT trajectories with DashScope.

The generator deliberately stores semantic OpenAI chat messages instead of
provider token IDs or rendered chat-template strings.  Every accepted sample
ends in one structured ``localization_finish`` call and is validated against
the SWE-Smith file/module/entity target before it enters the SFT dataset.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from openai import AsyncOpenAI

SCHEMA_VERSION = "codepin-teacher-trajectory-v1"
DEFAULT_INPUT = Path(
    "data/SWE-smith-code-search-sft-selection-6000/teacher_generation.parquet"
)
DEFAULT_SELECTION_INDEX = Path(
    "data/SWE-smith-code-search-sft-selection-6000/selection_index.parquet"
)
DEFAULT_OUTPUT = Path(
    "data/SWE-smith-code-search-teacher-qwen3.5-35b-a3b"
)
DEFAULT_REPO_CACHE = Path("data/.cache/swe_smith_repos")
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.5-35b-a3b"
DEFAULT_SYSTEM_PROMPT = Path(
    "src/prompts/templates/system_prompt_atomic_search.j2"
)
RETRY_STRATEGIES = (
    "",
    (
        "Prioritize literal identifiers, exception text, configuration keys, and "
        "API names from the issue; verify their definitions before following callers."
    ),
    (
        "Trace the behavior from tests or reproduction entry points through callers "
        "and callees, then distinguish modification targets from context-only files."
    ),
    (
        "Search for semantic synonyms as well as exact issue wording. Inspect nearby "
        "branches, assignments, and data-flow that could produce the described result."
    ),
    (
        "Broaden file discovery first, then narrow candidates by reading definitions "
        "and their direct consumers. Check whether more than one location is required."
    ),
    (
        "Independently reconstruct the faulty execution path. Be conservative about "
        "extra files and precise about existing class and function names."
    ),
)

MAX_GLOB_RESULTS = 100
MAX_GREP_RESULTS = 60
MAX_READ_LINES = 500
MAX_READ_CHARS = 16_000
MAX_FILE_BYTES = 5 * 1024 * 1024
IGNORED_PATH_PARTS = frozenset(
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


def canonical_tools() -> list[dict[str, Any]]:
    """Return the exact four-tool action space consumed by CodePin SFT."""
    return [
        {
            "type": "function",
            "function": {
                "name": "glob",
                "description": (
                    "Find repository files with a repository-relative glob pattern. "
                    "Results are sorted and limited to 100 files."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": 'Glob such as "**/*.py".',
                        },
                        "path": {
                            "type": "string",
                            "description": "Optional repository-relative directory.",
                        },
                    },
                    "required": ["pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "grep",
                "description": (
                    "Search repository text with a regular expression. Returns "
                    "up to 60 repository-relative paths, line numbers, and lines."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {
                            "type": "string",
                            "description": "Optional relative file or directory.",
                        },
                        "include": {
                            "type": "string",
                            "description": 'Optional glob such as "*.py".',
                        },
                    },
                    "required": ["pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": (
                    "Read a bounded line range from a repository-relative text file."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "localization_finish",
                "description": (
                    "Submit the final CodePin localization result and end the run. "
                    "This must be the only tool call in the final assistant turn."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "locations": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "file": {"type": "string"},
                                    "class_name": {"type": "string"},
                                    "function_name": {"type": "string"},
                                },
                                "required": ["file"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["locations"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--selection-index", type=Path, default=DEFAULT_SELECTION_INDEX
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-cache", type=Path, default=DEFAULT_REPO_CACHE)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path("/tmp/codepin-teacher-workspaces"),
        help="Disposable Linux-local worktree root; never used for saved results.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--repo-clone-concurrency", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--request-retries", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--instance-id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--prefetch-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--validate-workspaces-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--target-guided",
        action="store_true",
        help=(
            "Use gold locations as private teacher-only guidance to reconstruct "
            "compact grounded trajectories. The guidance is omitted from saved "
            "training messages and disclosed in generation metadata."
        ),
    )
    parser.add_argument(
        "--gold-finalize",
        action="store_true",
        help=(
            "After teacher-generated grounding searches, append the verified gold "
            "finish call locally. Intended for exceptionally large targets that "
            "cannot be copied reliably through a model completion."
        ),
    )
    parser.add_argument(
        "--accept-file-exact",
        action="store_true",
        help=(
            "Accept exact file localization even when a non-empty gold symbol level "
            "is not exact. The default requires every available gold level."
        ),
    )
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def attempt_temperature(base_temperature: float, attempt: int) -> float:
    """Vary retries enough to avoid repeatedly sampling the same failed trace."""
    schedule = (base_temperature, 0.0, 0.15, 0.3, 0.1, 0.4)
    return schedule[(attempt - 1) % len(schedule)]


def retry_strategy(attempt: int) -> str:
    return RETRY_STRATEGIES[(attempt - 1) % len(RETRY_STRATEGIES)]


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def compact_report(report: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "model",
        "available",
        "method",
        "visible_model_count",
        "expected",
        "saved_success",
        "usable",
        "train",
        "validation",
        "missing",
        "validation_error_count",
        "failure_files",
        "complete",
    )
    return {field: report[field] for field in fields if field in report}


def artifact_name(instance_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", instance_id).strip("._")
    digest = hashlib.sha256(instance_id.encode("utf-8")).hexdigest()[:12]
    return f"{safe[:120]}--{digest}.json"


def normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def validate_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("path must be a non-empty unpadded string")
    normalized = value.replace("\\", "/")
    parsed = PurePosixPath(normalized)
    if (
        parsed.is_absolute()
        or normalized.startswith("./")
        or ".." in parsed.parts
        or re.match(r"^[A-Za-z]:/", normalized)
    ):
        raise ValueError("path must be repository-relative without '.' or '..'")
    return parsed.as_posix()


def resolve_inside(root: Path, value: str | None, *, require_exists: bool = True) -> Path:
    relative = "." if value in {None, "", "."} else validate_relative_path(value)
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("path escapes the repository root") from exc
    if require_exists and not candidate.exists():
        raise ValueError(f"path does not exist: {relative}")
    return candidate


def render_user_prompt(problem_statement: str) -> str:
    return (
        "I have access to a Python code repository through the provided search "
        "tools. Consider the following issue description:\n\n"
        "<issue_description>\n"
        f"{problem_statement.strip()}\n"
        "</issue_description>\n\n"
        "Act as a code search agent and localize the specific existing files, "
        "classes, or functions that need modification to resolve the issue. Do "
        "not implement the fix. Use repository-relative paths and finish with "
        "exactly one localization_finish tool call."
    )


def normalize_locations(value: Any, workspace: Path | None = None) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("locations must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    signatures: set[tuple[str, str | None, str | None]] = set()
    for item in value:
        if not isinstance(item, dict):
            raise TypeError("each location must be an object")
        if set(item) - {"file", "class_name", "function_name"}:
            raise ValueError("location contains unsupported fields")
        file_path = validate_relative_path(item.get("file"))
        class_name = normalize_optional_text(item.get("class_name"))
        function_name = normalize_optional_text(item.get("function_name"))
        signature = (file_path, class_name, function_name)
        if signature in signatures:
            raise ValueError("locations contain a duplicate")
        signatures.add(signature)
        if workspace is not None and not (workspace / file_path).is_file():
            raise ValueError(f"location file does not exist: {file_path}")
        entry: dict[str, Any] = {"file": file_path}
        if class_name is not None:
            entry["class_name"] = class_name
        if function_name is not None:
            entry["function_name"] = function_name
        normalized.append(entry)
    return normalized


def parsed_levels(locations: Sequence[dict[str, Any]]) -> dict[str, set[str]]:
    files: set[str] = set()
    modules: set[str] = set()
    entities: set[str] = set()
    for location in locations:
        file_path = location["file"]
        class_name = location.get("class_name")
        function_name = location.get("function_name")
        files.add(file_path)
        if class_name:
            modules.add(f"{file_path}:{class_name}")
        elif function_name:
            modules.add(f"{file_path}:{function_name}")
        if class_name and function_name:
            entities.add(f"{file_path}:{class_name}.{function_name}")
        elif function_name:
            entities.add(f"{file_path}:{function_name}")
    return {"files": files, "modules": modules, "entities": entities}


def ground_truth_levels(row: dict[str, Any]) -> dict[str, set[str]]:
    files: set[str] = set()
    modules: set[str] = set()
    entities: set[str] = set()
    for change in row.get("file_changes") or row.get("target") or []:
        file_path = change.get("file")
        if file_path:
            files.add(file_path)
        details = change.get("changes") or {}
        for key in ("edited_modules", "added_modules"):
            modules.update(details.get(key) or [])
        for key in ("edited_entities", "added_entities"):
            entities.update(details.get(key) or [])
    return {"files": files, "modules": modules, "entities": entities}


def locations_from_ground_truth(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct exact structured locations from SWE-Smith target levels."""
    truth = ground_truth_levels(row)
    modules_by_file: dict[str, list[str]] = {}
    for value in truth["modules"]:
        file_path, symbol = value.split(":", 1)
        modules_by_file.setdefault(file_path, []).append(symbol)

    locations: list[dict[str, Any]] = []
    covered_files: set[str] = set()
    covered_modules: set[str] = set()
    for value in sorted(truth["entities"]):
        file_path, symbol = value.split(":", 1)
        containing = sorted(
            (
                module
                for module in modules_by_file.get(file_path, [])
                if symbol.startswith(module + ".")
            ),
            key=len,
            reverse=True,
        )
        entry: dict[str, Any] = {"file": file_path}
        if containing:
            class_name = containing[0]
            entry["class_name"] = class_name
            entry["function_name"] = symbol[len(class_name) + 1 :]
            covered_modules.add(f"{file_path}:{class_name}")
        else:
            entry["function_name"] = symbol
            covered_modules.add(f"{file_path}:{symbol}")
        locations.append(entry)
        covered_files.add(file_path)

    for value in sorted(truth["modules"] - covered_modules):
        file_path, symbol = value.split(":", 1)
        locations.append({"file": file_path, "class_name": symbol})
        covered_files.add(file_path)
    for file_path in sorted(truth["files"] - covered_files):
        locations.append({"file": file_path})

    normalized = normalize_locations(locations)
    if not score_locations(normalized, row)["perfect"]:
        raise ValueError("could not reconstruct exact structured gold locations")
    return normalized


def f1(predicted: set[str], truth: set[str]) -> float:
    if not truth:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    true_positive = len(predicted & truth)
    precision = true_positive / len(predicted)
    recall = true_positive / len(truth)
    return 2 * precision * recall / (precision + recall) if true_positive else 0.0


def score_locations(
    locations: Sequence[dict[str, Any]], row: dict[str, Any]
) -> dict[str, Any]:
    predicted = parsed_levels(locations)
    truth = ground_truth_levels(row)
    exact = {
        level: predicted[level] == truth[level]
        for level in ("files", "modules", "entities")
    }
    scores = {
        level: f1(predicted[level], truth[level])
        for level in ("files", "modules", "entities")
    }
    required_levels = ["files"] + [
        level for level in ("modules", "entities") if truth[level]
    ]
    return {
        "file_reward": scores["files"],
        "module_reward": scores["modules"],
        "entity_reward": scores["entities"],
        "multilevel_localization_f1_reward": sum(
            scores[level] for level in required_levels
        ),
        "required_levels": required_levels,
        "exact": exact,
        "perfect": all(exact[level] for level in required_levels),
        "predicted": {key: sorted(value) for key, value in predicted.items()},
        "ground_truth": {key: sorted(value) for key, value in truth.items()},
    }


def execute_glob(workspace: Path, arguments: dict[str, Any]) -> str:
    pattern = arguments.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("pattern must be a non-empty string")
    normalized = pattern.replace("\\", "/")
    if PurePosixPath(normalized).is_absolute() or ".." in PurePosixPath(normalized).parts:
        raise ValueError("pattern must be relative and cannot contain '..'")
    search_root = resolve_inside(workspace, arguments.get("path"))
    if not search_root.is_dir():
        raise ValueError("glob search path must be a directory")
    matches: list[str] = []
    for candidate in search_root.glob(normalized):
        try:
            relative = candidate.resolve().relative_to(workspace.resolve())
        except (OSError, ValueError):
            continue
        if not candidate.is_file() or IGNORED_PATH_PARTS.intersection(relative.parts):
            continue
        matches.append(relative.as_posix())
    matches = sorted(set(matches))
    truncated = len(matches) > MAX_GLOB_RESULTS
    matches = matches[:MAX_GLOB_RESULTS]
    if not matches:
        return f"No files found matching {pattern!r}."
    text = "\n".join(matches)
    if truncated:
        text += "\n\n[Results truncated to 100 files; narrow the search.]"
    return text


def execute_grep(workspace: Path, arguments: dict[str, Any]) -> str:
    pattern = arguments.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("pattern must be a non-empty string")
    target = resolve_inside(workspace, arguments.get("path"))
    command = [
        "rg",
        "--json",
        "--color=never",
        "--sort=path",
        f"--max-filesize={MAX_FILE_BYTES}",
        "--regexp",
        pattern,
    ]
    include = arguments.get("include")
    if include is not None:
        if not isinstance(include, str) or not include:
            raise ValueError("include must be a non-empty glob string")
        command.extend(["--glob", include])
    command.append(target.relative_to(workspace.resolve()).as_posix() or ".")
    process = subprocess.run(
        command,
        cwd=workspace,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    if process.returncode not in {0, 1}:
        raise ValueError(process.stderr.strip() or f"rg exited {process.returncode}")
    matches: list[str] = []
    truncated = False
    for raw_line in process.stdout.splitlines():
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
        if path_text is None or line_number is None or line_text is None:
            continue
        # ripgrep prefixes paths with "./" when the repository root is the
        # search target.  Tool-facing paths remain canonical and prefix-free.
        path_text = path_text.removeprefix("./")
        relative = resolve_inside(workspace, path_text).relative_to(
            workspace.resolve()
        )
        matches.append(
            f"{relative.as_posix()}:{line_number}:{line_text.rstrip(chr(10) + chr(13))}"
        )
        if len(matches) == MAX_GREP_RESULTS:
            truncated = True
            break
    text = "\n".join(matches) if matches else "No matches found."
    if truncated:
        text += "\n\n[Results truncated to 60 lines; narrow the search.]"
    return text


def execute_read_file(workspace: Path, arguments: dict[str, Any]) -> str:
    path_value = arguments.get("path")
    file_path = resolve_inside(workspace, path_value)
    if not file_path.is_file():
        raise ValueError(f"path is not a file: {path_value}")
    if file_path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("file exceeds the 5 MiB limit")
    with file_path.open("rb") as handle:
        if b"\x00" in handle.read(8192):
            raise ValueError("binary files are not supported")
    start = arguments.get("start_line", 1)
    end = arguments.get("end_line", start + 199)
    if not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start:
        raise ValueError("line range must contain positive ascending integers")
    if end - start + 1 > MAX_READ_LINES:
        raise ValueError("a read_file call can return at most 500 lines")
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if lines and start > len(lines):
        raise ValueError(f"start_line {start} exceeds the file's {len(lines)} lines")
    selected = lines[start - 1 : end]
    body_lines: list[str] = []
    char_count = 0
    truncated = False
    for number, line in enumerate(selected, start=start):
        remaining = MAX_READ_CHARS - char_count
        if remaining <= 0:
            truncated = True
            break
        if len(line) > remaining:
            line = line[:remaining]
            truncated = True
        body_lines.append(f"{number:6d}→{line}")
        char_count += len(line)
    relative = file_path.relative_to(workspace.resolve()).as_posix()
    actual_end = start + len(body_lines) - 1 if body_lines else 0
    text = (
        f"File: {relative} (lines {start}-{actual_end} of {len(lines)})\n"
        + "\n".join(body_lines)
    )
    if end < len(lines) or truncated:
        text += "\n\n[Content truncated; request a narrower or later range.]"
    return text


def execute_tool(
    workspace: Path, name: str, arguments: dict[str, Any]
) -> tuple[str, list[dict[str, Any]] | None, bool]:
    try:
        if name == "glob":
            return execute_glob(workspace, arguments), None, False
        if name == "grep":
            return execute_grep(workspace, arguments), None, False
        if name == "read_file":
            return execute_read_file(workspace, arguments), None, False
        if name == "localization_finish":
            locations = normalize_locations(arguments.get("locations"), workspace)
            return json.dumps(locations, ensure_ascii=False), locations, False
        raise ValueError(f"unsupported tool: {name}")
    except (OSError, subprocess.SubprocessError, TypeError, ValueError) as exc:
        return f"Error: {exc}", None, True


def serialize_assistant_message(message: Any) -> dict[str, Any]:
    tool_calls = []
    for tool_call in message.tool_calls or []:
        tool_calls.append(
            {
                "id": str(tool_call.id),
                "type": "function",
                "function": {
                    "name": str(tool_call.function.name),
                    "arguments": str(tool_call.function.arguments),
                },
            }
        )
    result: dict[str, Any] = {
        "role": "assistant",
        "content": message.content or "",
    }
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


def normalize_sft_messages(
    messages: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert OpenAI wire arguments to Qwen chat-template argument objects."""
    normalized = json.loads(json.dumps(messages, ensure_ascii=False))
    for message in normalized:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls", []):
            arguments = tool_call.get("function", {}).get("arguments")
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise TypeError("SFT tool-call arguments must be JSON objects")
            tool_call["function"]["arguments"] = arguments
    return normalized


@dataclass
class AttemptResult:
    accepted: bool
    record: dict[str, Any]
    reason: str


class FatalAPIError(RuntimeError):
    """A non-retryable API configuration or authentication failure."""


class RepositoryManager:
    def __init__(self, cache_dir: Path, temporary_root: Path):
        self.cache_dir = cache_dir.resolve()
        self.temporary_root = temporary_root.resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.temporary_root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, asyncio.Lock] = {}

    def mirror_path(self, repo: str) -> Path:
        safe = repo.replace("/", "__")
        return self.cache_dir / f"{safe}.git"

    def lock(self, repo: str) -> asyncio.Lock:
        return self._locks.setdefault(repo, asyncio.Lock())

    async def ensure_mirror(self, repo: str, retries: int = 4) -> Path:
        mirror = self.mirror_path(repo)
        async with self.lock(repo):
            if (mirror / "HEAD").is_file():
                return mirror
            temporary = mirror.with_name(mirror.name + ".partial")
            if temporary.exists():
                shutil.rmtree(temporary)
            url = f"https://github.com/{repo}.git"
            last_error = "unknown clone error"
            for attempt in range(retries):
                process = await asyncio.create_subprocess_exec(
                    "git",
                    "clone",
                    "--bare",
                    "--depth=1",
                    url,
                    str(temporary),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await process.communicate()
                if process.returncode == 0:
                    temporary.replace(mirror)
                    return mirror
                last_error = stderr.decode("utf-8", "replace").strip()
                if temporary.exists():
                    shutil.rmtree(temporary)
                await asyncio.sleep(min(2**attempt, 8))
            raise RuntimeError(f"could not clone {repo}: {last_error}")

    async def create_workspace(self, row: dict[str, Any], attempt: int) -> Path:
        repo = row["repo"]
        mirror = await self.ensure_mirror(repo)
        prefix = artifact_name(row["instance_id"]).removesuffix(".json")[:80]
        workspace = Path(
            tempfile.mkdtemp(
                prefix=f"{prefix}-a{attempt}-", dir=self.temporary_root
            )
        )
        shutil.rmtree(workspace)
        async with self.lock(repo):
            process = await asyncio.create_subprocess_exec(
                "git",
                f"--git-dir={mirror}",
                "worktree",
                "add",
                "--detach",
                str(workspace),
                "HEAD",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(stderr.decode("utf-8", "replace").strip())
        patch = row.get("patch") or ""
        if patch:
            process = await asyncio.create_subprocess_exec(
                "git",
                "-C",
                str(workspace),
                "apply",
                "--whitespace=nowarn",
                "-",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate(patch.encode("utf-8"))
            if process.returncode != 0:
                await self.remove_workspace(repo, workspace)
                raise RuntimeError(
                    "mutation patch did not apply: "
                    + stderr.decode("utf-8", "replace").strip()
                )
        return workspace

    async def remove_workspace(self, repo: str, workspace: Path) -> None:
        mirror = self.mirror_path(repo)
        async with self.lock(repo):
            if (mirror / "HEAD").is_file():
                process = await asyncio.create_subprocess_exec(
                    "git",
                    f"--git-dir={mirror}",
                    "worktree",
                    "remove",
                    "--force",
                    str(workspace),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await process.communicate()
        if workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)


class TeacherGenerator:
    def __init__(
        self,
        args: argparse.Namespace,
        client: AsyncOpenAI,
        repositories: RepositoryManager,
        system_prompt: str,
        selection_metadata: dict[str, dict[str, Any]],
    ):
        self.args = args
        self.client = client
        self.repositories = repositories
        self.system_prompt = system_prompt
        self.selection_metadata = selection_metadata
        self.tools = canonical_tools()
        self.finish_tool = [
            tool
            for tool in self.tools
            if tool.get("function", {}).get("name") == "localization_finish"
        ]
        self.search_tools = [
            tool
            for tool in self.tools
            if tool.get("function", {}).get("name") != "localization_finish"
        ]
        self.rate_limit_until = 0.0
        self.rate_limit_lock = asyncio.Lock()

    async def wait_for_rate_limit_window(self) -> None:
        async with self.rate_limit_lock:
            delay = self.rate_limit_until - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)

    async def extend_rate_limit_window(self, seconds: float = 65.0) -> None:
        async with self.rate_limit_lock:
            self.rate_limit_until = max(
                self.rate_limit_until, time.monotonic() + seconds
            )

    async def request(
        self,
        messages: list[dict[str, Any]],
        attempt: int,
        *,
        finalize: bool = False,
    ) -> Any:
        last_error: Exception | None = None
        for retry in range(self.args.request_retries):
            try:
                await self.wait_for_rate_limit_window()
                return await self.client.chat.completions.create(
                    model=self.args.model,
                    messages=messages,
                    tools=(
                        self.finish_tool
                        if finalize
                        else self.search_tools
                        if self.args.target_guided
                        else self.tools
                    ),
                    tool_choice=(
                        {
                            "type": "function",
                            "function": {"name": "localization_finish"},
                        }
                        if finalize
                        else "required"
                        if self.args.target_guided
                        else "auto"
                    ),
                    parallel_tool_calls=False,
                    temperature=attempt_temperature(
                        self.args.temperature, attempt
                    ),
                    max_tokens=self.args.max_output_tokens,
                    stream=False,
                    extra_body={"enable_thinking": False},
                )
            except Exception as exc:  # SDK exceptions vary across versions
                last_error = exc
                status_code = getattr(exc, "status_code", None)
                if status_code in {400, 401, 403, 404}:
                    raise FatalAPIError(
                        f"non-retryable API error ({status_code}): {exc}"
                    ) from exc
                if retry + 1 == self.args.request_retries:
                    break
                if status_code == 429:
                    # DashScope's insufficient_quota/token-limit response is a
                    # rolling TPM limit and normally clears within one minute.
                    await self.extend_rate_limit_window()
                    await self.wait_for_rate_limit_window()
                else:
                    await asyncio.sleep(min(2**retry + random.random(), 20))
        raise RuntimeError(f"API request failed after retries: {last_error}")

    async def run_attempt(
        self, row: dict[str, Any], attempt: int, prior_feedback: str | None
    ) -> AttemptResult:
        started_at = utc_now()
        started = time.monotonic()
        workspace: Path | None = None
        attempt_system_prompt = self.system_prompt
        strategy = retry_strategy(attempt)
        if prior_feedback or strategy:
            attempt_system_prompt += (
                "\n\nUse a fresh, independent search strategy and verify every "
                "candidate from source before finishing. The required answer "
                "must be exact. "
            )
            if prior_feedback:
                attempt_system_prompt += prior_feedback + " "
            if strategy:
                attempt_system_prompt += strategy + " "
            attempt_system_prompt += "Include only locations that need modification."
        api_system_prompt = attempt_system_prompt
        if self.args.target_guided:
            truth = {
                level: sorted(values)
                for level, values in ground_truth_levels(row).items()
            }
            api_system_prompt += (
                "\n\nPRIVATE TEACHER GUIDANCE (never shown to the student): "
                "The verified target levels are "
                + json.dumps(truth, ensure_ascii=False)
                + ". Reconstruct a concise, evidence-grounded search trajectory. "
                "Use at least one grep, glob, or read_file call to verify source "
                "evidence before submitting exactly these target levels with "
                "localization_finish. Preserve the correct class/function nesting."
            )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": api_system_prompt},
            {"role": "user", "content": render_user_prompt(row["problem_statement"])},
        ]

        def training_view() -> list[dict[str, Any]]:
            view = [
                message
                for message in json.loads(json.dumps(messages, ensure_ascii=False))
                if not (
                    message.get("role") == "user"
                    and str(message.get("content", "")).startswith(
                        "PRIVATE_TEACHER_FINALIZATION:"
                    )
                )
            ]
            view[0]["content"] = attempt_system_prompt
            return view

        usage = Counter()
        finish_locations: list[dict[str, Any]] | None = None
        finish_count = 0
        reason = "max_turns_without_finish"
        api_ids: list[str] = []
        try:
            workspace = await self.repositories.create_workspace(row, attempt)
            for turn in range(1, self.args.max_turns + 1):
                completion = await self.request(messages, attempt)
                api_ids.append(str(completion.id))
                if completion.usage is not None:
                    usage["prompt_tokens"] += int(completion.usage.prompt_tokens or 0)
                    usage["completion_tokens"] += int(
                        completion.usage.completion_tokens or 0
                    )
                    usage["total_tokens"] += int(completion.usage.total_tokens or 0)
                if not completion.choices:
                    reason = "api_returned_no_choices"
                    break
                assistant = serialize_assistant_message(
                    completion.choices[0].message
                )
                messages.append(assistant)
                tool_calls = assistant.get("tool_calls", [])
                if not tool_calls:
                    reason = (
                        "max_turns_without_finish"
                        if self.args.target_guided
                        else "assistant_returned_text_without_tool_call"
                    )
                    break
                names = [call["function"]["name"] for call in tool_calls]
                if "localization_finish" in names and (
                    len(tool_calls) != 1 or names[0] != "localization_finish"
                ):
                    reason = "finish_was_parallel_with_another_tool"
                    break
                turn_failed = False
                for tool_call in tool_calls:
                    name = tool_call["function"]["name"]
                    try:
                        arguments = json.loads(tool_call["function"]["arguments"])
                        if not isinstance(arguments, dict):
                            raise TypeError("tool arguments must decode to an object")
                    except (json.JSONDecodeError, TypeError) as exc:
                        observation = f"Error: invalid tool arguments: {exc}"
                        locations = None
                        is_error = True
                        reason = "invalid_tool_arguments"
                        turn_failed = True
                    else:
                        observation, locations, is_error = await asyncio.to_thread(
                            execute_tool, workspace, name, arguments
                        )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": name,
                            "content": observation,
                        }
                    )
                    if name == "localization_finish":
                        finish_count += 1
                        finish_locations = locations
                        if is_error:
                            reason = "invalid_finish_arguments"
                            turn_failed = True
                if turn_failed:
                    break
                if names == ["localization_finish"]:
                    reason = "finished"
                    break

            # Qwen may continue exploring even after it has enough evidence. Once
            # the bounded search budget is exhausted, request only the finish tool
            # so the saved trajectory ends in an explicit, trainable prediction.
            if reason == "max_turns_without_finish":
                if self.args.gold_finalize:
                    finish_locations = locations_from_ground_truth(row)
                    tool_call_id = (
                        "gold-finalize-"
                        + hashlib.sha256(
                            row["instance_id"].encode("utf-8")
                        ).hexdigest()[:16]
                    )
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": tool_call_id,
                                    "type": "function",
                                    "function": {
                                        "name": "localization_finish",
                                        "arguments": json.dumps(
                                            {"locations": finish_locations},
                                            ensure_ascii=False,
                                        ),
                                    },
                                }
                            ],
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": "localization_finish",
                            "content": json.dumps(
                                finish_locations, ensure_ascii=False
                            ),
                        }
                    )
                    finish_count += 1
                    reason = "finished"
                elif self.args.target_guided:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "PRIVATE_TEACHER_FINALIZATION: Submit exactly this "
                                "verified location array, without additions, removals, "
                                "or renamed fields: "
                                + json.dumps(
                                    locations_from_ground_truth(row),
                                    ensure_ascii=False,
                                )
                            ),
                        }
                    )
                if self.args.gold_finalize:
                    completion = None
                else:
                    completion = await self.request(messages, attempt, finalize=True)
                if completion is None:
                    pass
                else:
                    api_ids.append(str(completion.id))
                    if completion.usage is not None:
                        usage["prompt_tokens"] += int(completion.usage.prompt_tokens or 0)
                        usage["completion_tokens"] += int(
                            completion.usage.completion_tokens or 0
                        )
                        usage["total_tokens"] += int(completion.usage.total_tokens or 0)
                    if not completion.choices:
                        reason = "finalization_returned_no_choices"
                    else:
                        assistant = serialize_assistant_message(
                            completion.choices[0].message
                        )
                        messages.append(assistant)
                        tool_calls = assistant.get("tool_calls", [])
                        names = [
                            call.get("function", {}).get("name")
                            for call in tool_calls
                        ]
                        if names != ["localization_finish"]:
                            reason = "finalization_did_not_call_finish"
                        else:
                            tool_call = tool_calls[0]
                            try:
                                arguments = json.loads(
                                    tool_call["function"]["arguments"]
                                )
                                if not isinstance(arguments, dict):
                                    raise TypeError(
                                        "tool arguments must decode to an object"
                                    )
                            except (json.JSONDecodeError, TypeError) as exc:
                                observation = f"Error: invalid tool arguments: {exc}"
                                locations = None
                                is_error = True
                                reason = "invalid_finish_arguments"
                            else:
                                observation, locations, is_error = await asyncio.to_thread(
                                    execute_tool,
                                    workspace,
                                    "localization_finish",
                                    arguments,
                                )
                                reason = (
                                    "invalid_finish_arguments" if is_error else "finished"
                                )
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call["id"],
                                    "name": "localization_finish",
                                    "content": observation,
                                }
                            )
                            finish_count += 1
                            finish_locations = locations

            score = (
                score_locations(finish_locations, row)
                if finish_locations is not None
                else {
                    "file_reward": 0.0,
                    "module_reward": 0.0,
                    "entity_reward": 0.0,
                    "multilevel_localization_f1_reward": 0.0,
                    "required_levels": [],
                    "exact": {},
                    "perfect": False,
                    "predicted": {},
                    "ground_truth": {
                        key: sorted(value)
                        for key, value in ground_truth_levels(row).items()
                    },
                }
            )
            search_call_count = sum(
                call.get("function", {}).get("name")
                in {"glob", "grep", "read_file"}
                for message in messages
                for call in message.get("tool_calls", [])
            )
            structurally_valid = finish_count == 1 and finish_locations is not None
            if self.args.target_guided and search_call_count == 0:
                structurally_valid = False
                reason = "guided_finish_without_grounding_search"
            if structurally_valid and reason == "finished":
                accepted = (
                    score["exact"]["files"]
                    if self.args.accept_file_exact
                    else bool(score["perfect"])
                )
                if not accepted:
                    reason = "localization_not_exact"
            else:
                accepted = False
            metadata = self.selection_metadata.get(row["instance_id"], {})
            saved_messages = training_view()
            try:
                sft_messages = normalize_sft_messages(saved_messages)
            except (json.JSONDecodeError, TypeError, ValueError):
                # Failed attempts remain auditable, but malformed model calls must
                # never enter the trainable SFT view.
                sft_messages = []
            record = {
                "schema_version": SCHEMA_VERSION,
                "instance_id": row["instance_id"],
                "repo": row["repo"],
                "teacher_model": self.args.model,
                "api_protocol": "openai_chat_completions",
                "thinking_enabled": False,
                "attempt": attempt,
                "accepted": accepted,
                "acceptance_reason": "exact_gold_match" if accepted else reason,
                "total_reward": score["multilevel_localization_f1_reward"],
                "reward_dict": score,
                "structured_locations": finish_locations,
                "target": row.get("target"),
                "file_changes": row.get("file_changes"),
                "messages": saved_messages,
                "sft_messages": sft_messages,
                "tools": self.tools,
                "selection_metadata": metadata,
                "generation_metadata": {
                    "started_at": started_at,
                    "ended_at": utc_now(),
                    "duration_seconds": round(time.monotonic() - started, 3),
                    "assistant_turns": sum(
                        message["role"] == "assistant" for message in messages
                    ),
                    "tool_calls": sum(
                        len(message.get("tool_calls", [])) for message in messages
                    ),
                    "usage": dict(usage),
                    "api_response_ids": api_ids,
                    "retry_strategy": strategy,
                    "generation_mode": (
                        "target_guided_teacher_reconstruction_gold_finalized"
                        if self.args.gold_finalize
                        else
                        "target_guided_teacher_reconstruction"
                        if self.args.target_guided
                        else "autonomous_teacher_rollout"
                    ),
                    "private_target_guidance_omitted_from_training_messages": bool(
                        self.args.target_guided
                    ),
                    "temperature": attempt_temperature(
                        self.args.temperature, attempt
                    ),
                },
            }
            return AttemptResult(accepted=accepted, record=record, reason=reason)
        except Exception as exc:  # noqa: BLE001 - persist per-sample failures
            fatal = isinstance(exc, FatalAPIError)
            record = {
                "schema_version": SCHEMA_VERSION,
                "instance_id": row.get("instance_id"),
                "repo": row.get("repo"),
                "teacher_model": self.args.model,
                "attempt": attempt,
                "accepted": False,
                "acceptance_reason": "fatal_api_error" if fatal else "runtime_error",
                "fatal": fatal,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "messages": training_view(),
                "generation_metadata": {
                    "started_at": started_at,
                    "ended_at": utc_now(),
                    "duration_seconds": round(time.monotonic() - started, 3),
                },
            }
            return AttemptResult(
                False, record, "fatal_api_error" if fatal else "runtime_error"
            )
        finally:
            if workspace is not None:
                await self.repositories.remove_workspace(row["repo"], workspace)

    async def generate_one(self, row: dict[str, Any]) -> AttemptResult:
        name = artifact_name(row["instance_id"])
        success_path = self.args.output / "trajectories" / name
        if success_path.is_file():
            try:
                existing = json.loads(success_path.read_text(encoding="utf-8"))
                if (
                    existing.get("accepted")
                    and existing.get("schema_version") == SCHEMA_VERSION
                    and existing.get("teacher_model") == self.args.model
                    and existing.get("thinking_enabled") is False
                    and not validate_saved_record(existing)
                ):
                    return AttemptResult(True, existing, "already_complete")
            except (OSError, json.JSONDecodeError):
                pass

        prior_feedback: str | None = None
        last: AttemptResult | None = None
        for attempt in range(1, self.args.max_attempts + 1):
            result = await self.run_attempt(row, attempt, prior_feedback)
            attempt_path = (
                self.args.output
                / "attempts"
                / artifact_name(row["instance_id"]).removesuffix(".json")
                / f"attempt-{attempt}.json"
            )
            atomic_write_json(attempt_path, result.record)
            if result.accepted:
                atomic_write_json(success_path, result.record)
                failure_path = self.args.output / "failures" / name
                if failure_path.exists():
                    failure_path.unlink()
                return result
            last = result
            if result.record.get("fatal"):
                break
            reward = result.record.get("reward_dict", {})
            exact = reward.get("exact", {})
            if exact:
                expected = reward.get("ground_truth", {})
                predicted = reward.get("predicted", {})
                count_feedback = []
                for level, is_exact in exact.items():
                    if is_exact:
                        continue
                    count_feedback.append(
                        f"{level}: expected {len(expected.get(level, []))}, "
                        f"submitted {len(predicted.get(level, []))}"
                    )
                prior_feedback = (
                    "The previous result was not exact. Evaluator count feedback "
                    "(without revealing target names): "
                    + "; ".join(count_feedback)
                    + ". Re-check omissions and remove context-only extras."
                )
            else:
                prior_feedback = (
                    f"The previous attempt failed because {result.reason}."
                )

        assert last is not None
        atomic_write_json(self.args.output / "failures" / name, last.record)
        return last


def load_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def load_selection_metadata(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    return {
        row["instance_id"]: {
            key: value
            for key, value in row.items()
            if key
            in {
                "difficulty",
                "selection_bucket",
                "mutation_type",
                "boundary_score",
                "mean_normalized_reward",
                "perfect_rate",
            }
        }
        for row in load_rows(path)
    }


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.instance_id:
        selected = [row for row in rows if row["instance_id"] == args.instance_id]
        if not selected:
            raise ValueError(f"unknown --instance-id {args.instance_id!r}")
        return selected
    if args.offset < 0:
        raise ValueError("--offset must be non-negative")
    end = None if args.limit is None else args.offset + args.limit
    return rows[args.offset:end]


async def prefetch_repositories(
    manager: RepositoryManager, rows: Sequence[dict[str, Any]], concurrency: int
) -> None:
    repos = sorted({row["repo"] for row in rows})
    semaphore = asyncio.Semaphore(max(1, concurrency))
    completed = 0
    lock = asyncio.Lock()

    async def fetch(repo: str) -> None:
        nonlocal completed
        async with semaphore:
            await manager.ensure_mirror(repo)
        async with lock:
            completed += 1
            print(f"repositories {completed}/{len(repos)}", flush=True)

    await asyncio.gather(*(fetch(repo) for repo in repos))


async def preflight_api(
    client: AsyncOpenAI,
    model: str,
    output_path: Path,
    api_key: str,
) -> dict[str, Any]:
    """Fail before a large run when the key/endpoint cannot access the model."""
    started_at = utc_now()
    try:
        page = await client.models.list()
        model_ids = sorted({str(item.id) for item in page.data})
        if model not in model_ids:
            raise FatalAPIError(
                f"model {model!r} is not available to this key/endpoint; "
                f"available models: {', '.join(model_ids)}"
            )
        method = "models.list"
    except FatalAPIError:
        raise
    except Exception as exc:
        status_code = getattr(exc, "status_code", None)
        if status_code in {401, 403}:
            raise FatalAPIError(
                f"API authentication/authorization failed ({status_code}): {exc}"
            ) from exc
        # Some OpenAI-compatible providers omit /models. A one-token request is
        # the smallest reliable capability check in that case.
        try:
            await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply with OK."}],
                temperature=0.0,
                max_tokens=1,
                stream=False,
                extra_body={"enable_thinking": False},
            )
        except Exception as chat_exc:
            safe_error = str(chat_exc).replace(api_key, "<redacted>")
            raise FatalAPIError(f"model capability preflight failed: {safe_error}") from chat_exc
        model_ids = [model]
        method = "one_token_chat_fallback"

    report = {
        "generated_at": utc_now(),
        "started_at": started_at,
        "model": model,
        "available": True,
        "method": method,
        "visible_model_count": len(model_ids),
        "visible_models": model_ids,
        "api_key_family": (
            "plan" if api_key.startswith("sk-sp-") else "general_or_workspace"
        ),
    }
    atomic_write_json(output_path, report)
    return report


async def validate_workspaces(
    manager: RepositoryManager,
    rows: Sequence[dict[str, Any]],
    concurrency: int,
    report_path: Path,
) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    progress_lock = asyncio.Lock()
    failures: dict[str, str] = {}
    completed = 0

    async def validate(row: dict[str, Any]) -> None:
        nonlocal completed
        workspace: Path | None = None
        try:
            async with semaphore:
                workspace = await manager.create_workspace(row, 0)
                missing = [
                    path
                    for path in sorted(ground_truth_levels(row)["files"])
                    if not (workspace / path).is_file()
                ]
                if missing:
                    raise RuntimeError(f"gold files missing after patch: {missing}")
        except Exception as exc:  # noqa: BLE001 - validate every workspace
            failures[row["instance_id"]] = f"{type(exc).__name__}: {exc}"
        finally:
            if workspace is not None:
                await manager.remove_workspace(row["repo"], workspace)
            async with progress_lock:
                completed += 1
                if completed % 100 == 0 or completed == len(rows):
                    print(
                        f"workspaces {completed}/{len(rows)} "
                        f"failed={len(failures)}",
                        flush=True,
                    )

    await asyncio.gather(*(validate(row) for row in rows))
    report = {
        "generated_at": utc_now(),
        "expected": len(rows),
        "validated": len(rows) - len(failures),
        "failed": len(failures),
        "failures": dict(sorted(failures.items())),
        "complete": not failures,
    }
    atomic_write_json(report_path, report)
    return report


def read_success_records(output: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((output / "trajectories").glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("accepted") and record.get("schema_version") == SCHEMA_VERSION:
            records.append(record)
    return records


def validate_saved_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    messages = record.get("sft_messages")
    tools = record.get("tools")
    if not isinstance(messages, list) or not messages:
        errors.append("missing_messages")
        return errors
    if tools != canonical_tools():
        errors.append("tool_schema_mismatch")
    finish_calls = []
    call_ids: set[str] = set()
    observations: set[str] = set()
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {
            "system",
            "user",
            "assistant",
            "tool",
        }:
            errors.append("invalid_message")
            continue
        if "reasoning_content" in message:
            errors.append("reasoning_content_present")
        if message.get("role") == "assistant":
            for call in message.get("tool_calls", []):
                call_id = call.get("id")
                if not call_id or call_id in call_ids:
                    errors.append("duplicate_or_missing_call_id")
                call_ids.add(call_id)
                if call.get("function", {}).get("name") == "localization_finish":
                    finish_calls.append((message, call))
        if message.get("role") == "tool":
            observations.add(message.get("tool_call_id"))
    if len(finish_calls) != 1:
        errors.append("finish_call_count")
    else:
        message, _ = finish_calls[0]
        if len(message.get("tool_calls", [])) != 1:
            errors.append("invalid_finish_turn")
    if call_ids != observations:
        errors.append("tool_call_observation_mismatch")
    reward = record.get("reward_dict", {})
    if not reward.get("perfect"):
        errors.append("not_gold_exact")
    if not record.get("accepted"):
        errors.append("not_accepted")
    return sorted(set(errors))


def write_parquet_snapshot(path: Path, records: list[dict[str, Any]]) -> None:
    """Write an audit parquet without weakening the nested JSONL training contract."""
    rows = [
        {
            "instance_id": record["instance_id"],
            "repo": record["repo"],
            "teacher_model": record["teacher_model"],
            "messages_json": json.dumps(record["sft_messages"], ensure_ascii=False),
            "tools_json": json.dumps(record["tools"], ensure_ascii=False),
            "structured_locations_json": json.dumps(
                record["structured_locations"], ensure_ascii=False
            ),
            "reward_json": json.dumps(record["reward_dict"], ensure_ascii=False),
            "selection_bucket": record.get("selection_metadata", {}).get(
                "selection_bucket"
            ),
            "difficulty": record.get("selection_metadata", {}).get("difficulty"),
        }
        for record in records
    ]
    schema = pa.schema(
        [
            pa.field("instance_id", pa.string()),
            pa.field("repo", pa.string()),
            pa.field("teacher_model", pa.string()),
            pa.field("messages_json", pa.string()),
            pa.field("tools_json", pa.string()),
            pa.field("structured_locations_json", pa.string()),
            pa.field("reward_json", pa.string()),
            pa.field("selection_bucket", pa.string()),
            pa.field("difficulty", pa.string()),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)


def stratified_split(
    records: Sequence[dict[str, Any]], seed: int, validation_ratio: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically preserve selection bucket × difficulty proportions."""
    if not records or validation_ratio <= 0:
        return list(records), []
    validation_size = round(len(records) * validation_ratio)
    validation_size = max(1, min(validation_size, len(records) - 1))
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        metadata = record.get("selection_metadata", {})
        key = (
            str(metadata.get("selection_bucket", "unknown")),
            str(metadata.get("difficulty", "unknown")),
        )
        groups.setdefault(key, []).append(record)

    exact_quotas = {
        key: len(group) * validation_ratio for key, group in groups.items()
    }
    quotas = {key: int(value) for key, value in exact_quotas.items()}
    remaining = validation_size - sum(quotas.values())
    quota_order = sorted(
        groups,
        key=lambda key: (-(exact_quotas[key] - quotas[key]), key),
    )
    for key in quota_order[:remaining]:
        quotas[key] += 1

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda record: record["instance_id"])
        group_seed = int.from_bytes(
            hashlib.sha256(
                f"{seed}:{key[0]}:{key[1]}".encode()
            ).digest()[:8],
            "big",
        )
        random.Random(group_seed).shuffle(group)
        validation.extend(group[: quotas[key]])
        train.extend(group[quotas[key] :])
    random.Random(seed).shuffle(train)
    random.Random(seed + 1).shuffle(validation)
    return train, validation


def aggregate(output: Path, expected_rows: Sequence[dict[str, Any]], seed: int, validation_ratio: float) -> dict[str, Any]:
    records = read_success_records(output)
    expected_ids = {row["instance_id"] for row in expected_rows}
    by_id = {record["instance_id"]: record for record in records}
    duplicate_success = len(records) - len(by_id)
    validation_errors = {
        instance_id: errors
        for instance_id, record in by_id.items()
        if (errors := validate_saved_record(record))
    }
    usable = [
        record
        for instance_id, record in by_id.items()
        if instance_id in expected_ids and instance_id not in validation_errors
    ]
    usable.sort(key=lambda record: record["instance_id"])
    train, validation = stratified_split(
        usable, seed=seed, validation_ratio=validation_ratio
    )

    def sft_row(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "instance_id": record["instance_id"],
            "messages": record["sft_messages"],
            "tools": record["tools"],
            "reward": record["total_reward"],
            "teacher_model": record["teacher_model"],
            "source": str(
                output / "trajectories" / artifact_name(record["instance_id"])
            ),
        }

    dataset = output / "sft_dataset"
    atomic_write_jsonl(dataset / "all.jsonl", (sft_row(row) for row in usable))
    atomic_write_jsonl(dataset / "train.jsonl", (sft_row(row) for row in train))
    atomic_write_jsonl(
        dataset / "validation.jsonl", (sft_row(row) for row in validation)
    )
    write_parquet_snapshot(output / "trajectories-audit.parquet", usable)

    failure_files = list((output / "failures").glob("*.json"))
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "expected": len(expected_ids),
        "saved_success": len(by_id),
        "usable": len(usable),
        "train": len(train),
        "validation": len(validation),
        "missing": len(expected_ids - set(by_id)),
        "missing_instance_ids": sorted(expected_ids - set(by_id))[:100],
        "unexpected_success": len(set(by_id) - expected_ids),
        "duplicate_success": duplicate_success,
        "validation_error_count": len(validation_errors),
        "validation_errors": dict(sorted(validation_errors.items())[:100]),
        "failure_files": len(failure_files),
        "teacher_model_counts": dict(
            Counter(record["teacher_model"] for record in usable)
        ),
        "thinking_enabled_counts": dict(
            Counter(str(record["thinking_enabled"]) for record in usable)
        ),
        "generation_mode_counts": dict(
            Counter(
                record.get("generation_metadata", {}).get(
                    "generation_mode", "autonomous_teacher_rollout_legacy"
                )
                for record in usable
            )
        ),
        "selection_bucket_counts": dict(
            Counter(
                record.get("selection_metadata", {}).get("selection_bucket", "unknown")
                for record in usable
            )
        ),
        "difficulty_counts": dict(
            Counter(
                record.get("selection_metadata", {}).get("difficulty", "unknown")
                for record in usable
            )
        ),
        "train_selection_bucket_counts": dict(
            Counter(
                record.get("selection_metadata", {}).get(
                    "selection_bucket", "unknown"
                )
                for record in train
            )
        ),
        "validation_selection_bucket_counts": dict(
            Counter(
                record.get("selection_metadata", {}).get(
                    "selection_bucket", "unknown"
                )
                for record in validation
            )
        ),
        "train_difficulty_counts": dict(
            Counter(
                record.get("selection_metadata", {}).get("difficulty", "unknown")
                for record in train
            )
        ),
        "validation_difficulty_counts": dict(
            Counter(
                record.get("selection_metadata", {}).get("difficulty", "unknown")
                for record in validation
            )
        ),
        "complete": (
            len(usable) == len(expected_ids)
            and not validation_errors
            and not (expected_ids - set(by_id))
        ),
    }
    atomic_write_json(output / "report.json", report)
    return report


async def async_main(args: argparse.Namespace) -> int:
    if args.concurrency < 1 or args.repo_clone_concurrency < 1:
        raise ValueError("concurrency values must be positive")
    if args.max_turns < 1 or args.max_attempts < 1 or args.request_retries < 1:
        raise ValueError("turn and retry values must be positive")
    if not 0 <= args.validation_ratio < 1:
        raise ValueError("--validation-ratio must be in [0, 1)")
    all_rows = load_rows(args.input)
    rows = select_rows(all_rows, args)
    args.output.mkdir(parents=True, exist_ok=True)
    repositories = RepositoryManager(args.repo_cache, args.workspace_root)

    if args.aggregate_only:
        report = aggregate(args.output, all_rows, args.seed, args.validation_ratio)
        print(json.dumps(compact_report(report), ensure_ascii=False, indent=2), flush=True)
        return 0 if report["complete"] else 2

    if args.prefetch_only or args.validate_workspaces_only:
        await prefetch_repositories(
            repositories, rows, args.repo_clone_concurrency
        )
    if args.prefetch_only:
        return 0
    if args.validate_workspaces_only:
        report = await validate_workspaces(
            repositories,
            rows,
            args.concurrency,
            args.output / "workspace-validation.json",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0 if report["complete"] else 2

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"environment variable {args.api_key_env!r} is not configured"
        )
    if api_key.startswith("sk-sp-") and args.base_url.rstrip("/") == DEFAULT_BASE_URL:
        error = (
            "A sk-sp- Token/Coding Plan key cannot use the pay-as-you-go "
            "DashScope endpoint. Use the matching plan endpoint, or provide a "
            f"general sk-/sk-ws- key for {args.model}."
        )
        atomic_write_json(
            args.output / "api-preflight.json",
            {
                "generated_at": utc_now(),
                "model": args.model,
                "available": False,
                "api_key_family": "plan",
                "error": error,
            },
        )
        raise RuntimeError(error)
    system_prompt = args.system_prompt.read_text(encoding="utf-8").strip()
    selection_metadata = load_selection_metadata(args.selection_index)
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=args.base_url,
        timeout=args.request_timeout,
        max_retries=0,
    )
    try:
        preflight_report = await preflight_api(
            client,
            args.model,
            args.output / "api-preflight.json",
            api_key,
        )
        print(
            json.dumps(compact_report(preflight_report), ensure_ascii=False, indent=2),
            flush=True,
        )
    except Exception as exc:
        safe_error = str(exc).replace(api_key, "<redacted>")
        atomic_write_json(
            args.output / "api-preflight.json",
            {
                "generated_at": utc_now(),
                "model": args.model,
                "available": False,
                "api_key_family": (
                    "plan"
                    if api_key.startswith("sk-sp-")
                    else "general_or_workspace"
                ),
                "error_type": type(exc).__name__,
                "error": safe_error,
            },
        )
        await client.close()
        raise
    if args.preflight_only:
        await client.close()
        return 0

    await prefetch_repositories(
        repositories, rows, args.repo_clone_concurrency
    )
    generator = TeacherGenerator(
        args, client, repositories, system_prompt, selection_metadata
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    progress_lock = asyncio.Lock()
    counters = Counter()

    async def worker(row: dict[str, Any]) -> None:
        async with semaphore:
            result = await generator.generate_one(row)
        async with progress_lock:
            counters["processed"] += 1
            counters["accepted" if result.accepted else "failed"] += 1
            print(
                f"samples {counters['processed']}/{len(rows)} "
                f"accepted={counters['accepted']} failed={counters['failed']} "
                f"last={row['instance_id']} reason={result.reason}",
                flush=True,
            )
        if result.reason == "fatal_api_error":
            raise FatalAPIError(str(result.record.get("error", "fatal API error")))

    try:
        await asyncio.gather(*(worker(row) for row in rows))
    finally:
        await client.close()
    report = aggregate(args.output, all_rows, args.seed, args.validation_ratio)
    print(json.dumps(compact_report(report), ensure_ascii=False, indent=2), flush=True)
    return 0 if report["complete"] else 2


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
