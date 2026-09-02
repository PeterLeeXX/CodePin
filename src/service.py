"""Bounded localization batches and content-addressed, process-local result cache."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.context import bounded_context
from src.rollout import run_localization


class LocalizationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repository: str = Field(min_length=1)
    issue: str = Field(min_length=1, max_length=60000)
    max_context_chars: int = Field(default=12000, ge=1, le=30000)
    max_context_lines: int = Field(default=160, ge=1, le=500)


@dataclass(frozen=True)
class ServiceConfig:
    repository_root: Path
    model: str = "openai/codepin"
    base_url: str = "http://127.0.0.1:8000/v1"
    max_turns: int = 8
    max_tokens: int = 2048
    concurrency: int = 4
    cache_size: int = 64
    cache_ttl: float = 300
    deployment_file: Path | None = None

    def __post_init__(self):
        if not self.repository_root.is_dir():
            raise ValueError("repository_root must be an existing directory")
        if not 1 <= self.concurrency <= 32 or not 1 <= self.max_turns <= 32:
            raise ValueError("concurrency and max_turns must be in 1..32")
        if self.max_tokens < 1 or self.cache_size < 0 or self.cache_ttl <= 0:
            raise ValueError("invalid token or cache limits")
        if self.cache_size and (
            not self.deployment_file or not self.deployment_file.is_file()
        ):
            raise ValueError("result caching requires the server's deployment file")


def tree_digest(root: Path) -> str:
    """Hash bytes, names and link targets, including dirty/untracked/ignored files.

    No mtime shortcut: same-size edits and restored timestamps must invalidate.
    External links are never followed. Internal link targets are hashed in-tree.
    """
    digest = hashlib.sha256()
    for parent, dirs, files in os.walk(root):
        dirs.sort()
        for name in sorted(dirs + files):
            path = Path(parent) / name
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            if path.is_symlink():
                digest.update(b"link\0" + os.readlink(path).encode() + b"\0")
            elif path.is_file():
                digest.update(b"file\0")
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                digest.update(b"\0")
    return digest.hexdigest()


class ResultCache:
    def __init__(self, size: int, ttl: float):
        self.size, self.ttl = size, ttl
        self.entries: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: str) -> dict | None:
        with self.lock:
            entry = self.entries.pop(key, None)
            if entry is None or time.monotonic() - entry[0] >= self.ttl:
                return None
            self.entries[key] = entry
            return copy.deepcopy(entry[1])

    def put(self, key: str, value: dict) -> None:
        if not self.size or value["status"] != "ok":
            return
        with self.lock:
            self.entries.pop(key, None)
            self.entries[key] = (time.monotonic(), copy.deepcopy(value))
            while len(self.entries) > self.size:
                self.entries.popitem(last=False)


class LocalizationService:
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.slots = asyncio.Semaphore(config.concurrency)
        self.cache = ResultCache(config.cache_size, config.cache_ttl)

    def repository(self, value: str) -> Path:
        root = self.config.repository_root.resolve()
        path = (root / value).resolve()
        path.relative_to(root)
        if not path.is_dir():
            raise ValueError("repository must be a directory under repository_root")
        return path

    def cache_key(self, request: LocalizationRequest, snapshot: str) -> str:
        config = asdict(self.config)
        deployment = (
            self.config.deployment_file.read_text(encoding="utf-8")
            if self.config.deployment_file
            else "uncached"
        )
        # Changes to prompts, schemas, budgets or implementation invalidate results.
        source_root = Path(__file__).parent
        code = hashlib.sha256()
        for path in sorted([*source_root.rglob("*.py"), *source_root.rglob("*.j2")]):
            code.update(path.relative_to(source_root).as_posix().encode())
            code.update(path.read_bytes())
        payload = [request.model_dump(), snapshot, config, deployment, code.hexdigest()]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()

    async def localize(
        self, request: LocalizationRequest, *, purpose: str = "serving"
    ) -> dict:
        if purpose not in {"serving", "rollout"}:
            raise ValueError("purpose must be serving or rollout")
        if purpose == "rollout" and self.config.cache_size:
            raise ValueError("training rollout must disable result caching")
        root = self.repository(request.repository)
        if not request.issue.strip():
            raise ValueError("issue cannot be blank")
        await asyncio.wait_for(self.slots.acquire(), timeout=60)
        try:
            task = asyncio.create_task(asyncio.to_thread(self._localize, request, root))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                # The SDK runs synchronously; keep its slot until it exits.
                await task
                raise
        finally:
            self.slots.release()

    def _localize(self, request: LocalizationRequest, root: Path) -> dict:
        snapshot = tree_digest(root)
        key = self.cache_key(request, snapshot)
        if self.config.cache_size and (cached := self.cache.get(key)):
            cached["cache_hit"] = True
            return cached
        result = run_localization(
            {"problem_statement": request.issue},
            root,
            model=self.config.model,
            base_url=self.config.base_url,
            max_turns=self.config.max_turns,
            max_tokens=self.config.max_tokens,
        )
        context = []
        if result["status"] == "ok":
            context = bounded_context(
                root,
                result["structured_locations"],
                request.max_context_chars,
                request.max_context_lines,
            )
        if snapshot != tree_digest(root) or key != self.cache_key(request, snapshot):
            result["errors"].append("repository_or_deployment_changed_during_run")
            result["status"] = "error"
            result["structured_locations"] = None
            context = []
        response = {
            "status": result["status"],
            "locations": result["structured_locations"] or [],
            "context": context,
            "metrics": result["metrics"],
            "errors": result["errors"],
            "snapshot": snapshot,
            "cache_hit": False,
        }
        self.cache.put(key, response)
        return response

    async def batch(self, requests: list[LocalizationRequest]) -> list[dict]:
        if not 1 <= len(requests) <= 32:
            raise ValueError("a batch must contain 1..32 tasks")
        # Requests enter vLLM's native continuous batch scheduler concurrently.
        # Every trajectory retains separate tools, workspace and token history.
        results = await asyncio.gather(
            *(self.localize(r) for r in requests), return_exceptions=True
        )
        return [
            {"status": "error", "errors": [str(r)]} if isinstance(r, Exception) else r
            for r in results
        ]
