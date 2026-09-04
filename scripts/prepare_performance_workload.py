"""Pin and materialize the fixed CodePin serving performance workload."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from src.build_swe_smith_code_search import analyze_source
from src.data_pipeline import clean_tasks, load_rows
from src.performance import source_manifest
from src.service import tree_digest
from src.utils.instance import CACHE_ROOT, clone_instance

GIT_ENV = {
    **os.environ,
    "GIT_HTTP_LOW_SPEED_LIMIT": "1024",
    "GIT_HTTP_LOW_SPEED_TIME": "30",
}
GIT_ATTEMPTS = 3


def git_run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
        env=GIT_ENV,
    )


def git_output(*args: str) -> str:
    return git_run(*args).stdout.strip()


def fetch_commit(mirror: Path, commit: str) -> None:
    error: subprocess.CalledProcessError | subprocess.TimeoutExpired | None = None
    for attempt in range(GIT_ATTEMPTS):
        try:
            git_run(
                "--git-dir",
                str(mirror),
                "fetch",
                "--depth=1",
                "origin",
                commit,
            )
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            error = exc
            if attempt + 1 < GIT_ATTEMPTS:
                time.sleep(2**attempt)
    assert error is not None
    raise error


def resolve_commit(repo: str, declared: str | None) -> str:
    if declared:
        commit = declared
    else:
        output = git_output("ls-remote", f"https://github.com/{repo}.git", "HEAD")
        commit = output.split()[0] if output else ""
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise RuntimeError(
            f"repository did not resolve to an immutable commit: {commit}"
        )
    return commit


def ensure_commit_mirror(repo: str, commit: str) -> Path:
    """Cache only the exact benchmark commit instead of the repository history."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    name = repo.replace("/", "__")
    mirror = CACHE_ROOT / f"{name}.git"
    lock_path = CACHE_ROOT / f"{name}.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if mirror.is_dir():
            try:
                git_run(
                    "--git-dir", str(mirror), "cat-file", "-e", f"{commit}^{{commit}}"
                )
                return mirror
            except subprocess.CalledProcessError:
                fetch_commit(mirror, commit)
                return mirror

        temporary = CACHE_ROOT / f"{name}.{uuid.uuid4().hex}.tmp"
        try:
            git_run("init", "--bare", str(temporary))
            git_run(
                "--git-dir",
                str(temporary),
                "remote",
                "add",
                "origin",
                f"https://github.com/{repo}.git",
            )
            fetch_commit(temporary, commit)
            git_run(
                "--git-dir",
                str(temporary),
                "update-ref",
                "refs/heads/codepin",
                commit,
            )
            git_run(
                "--git-dir",
                str(temporary),
                "symbolic-ref",
                "HEAD",
                "refs/heads/codepin",
            )
            temporary.replace(mirror)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return mirror


def audit_targets(task: dict, workspace: Path) -> dict:
    """Expose unattainable addition labels without silently changing the oracle."""
    missing_added, checked = [], 0
    for change in task["file_changes"]:
        path = (workspace / change["file"]).resolve()
        path.relative_to(workspace.resolve())
        analysis = analyze_source(path.read_text(encoding="utf-8"), change["file"])
        entities = {f.entity for f in analysis.functions} | {
            f.entity for _, methods in analysis.classes for f in methods
        }
        modules = {f.module for f in analysis.functions} | {
            cls.module for cls, _ in analysis.classes
        }
        for key in (
            "edited_entities",
            "added_entities",
            "edited_modules",
            "added_modules",
        ):
            for target in (change.get("changes") or {}).get(key) or []:
                checked += 1
                symbol = target.split(":", 1)[1]
                if symbol in (entities if key.endswith("entities") else modules):
                    continue
                if key.startswith("edited_"):
                    raise ValueError(
                        f"existing target not found: {task['instance_id']}: {target}"
                    )
                missing_added.append({"field": key, "target": target})
    return {
        "checked_symbol_references": checked,
        "missing_added_targets": missing_added,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.workspace_root.exists():
        raise FileExistsError("output and workspace-root must both be new paths")

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    dataset_digest = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    if spec.get("dataset_sha256") not in (None, dataset_digest):
        raise ValueError(
            "dataset content differs from the frozen workload specification"
        )
    rows = load_rows(args.dataset)
    selected = []
    split_by_index = {}
    for split in ("tuning", "validation"):
        for index in spec[f"{split}_indices"]:
            if index in split_by_index:
                raise ValueError(f"dataset index {index} appears in both splits")
            split_by_index[index] = split
            selected.append((index, dict(rows[index])))

    args.output.mkdir(parents=True)
    args.workspace_root.mkdir(parents=True)
    prepared = []
    for index, task in selected:
        started = time.monotonic()
        print(
            json.dumps(
                {"event": "prepare_start", "dataset_index": index, "repo": task["repo"]}
            ),
            flush=True,
        )
        pinned = spec.get("revisions", {}).get(task["repo"])
        if pinned and task.get("base_commit") not in (None, "", pinned):
            raise ValueError(f"conflicting revision for {task['repo']}")
        commit = resolve_commit(task["repo"], pinned or task.get("base_commit"))
        ensure_commit_mirror(task["repo"], commit)
        task["base_commit"] = commit
        cleaned, report = clean_tasks([task])
        if len(cleaned) != 1:
            raise ValueError(f"selected task failed cleaning: {report}")
        task = cleaned[0]
        ok, workspace = clone_instance(
            task["repo"],
            commit,
            task["instance_id"],
            args.workspace_root,
            task.get("patch") if task.get("use_patch") else None,
        )
        if not ok or workspace is None:
            raise RuntimeError(f"failed to prepare {task['instance_id']}")
        targets = audit_targets(task, workspace)
        source = source_manifest(workspace)
        manifest_path = args.output / f"source-{index}.json"
        manifest_path.write_text(json.dumps(source, indent=2), encoding="utf-8")
        digest_started = time.monotonic()
        snapshot = tree_digest(workspace)
        digest_seconds = time.monotonic() - digest_started
        task.update(
            benchmark_split=split_by_index[index],
            dataset_index=index,
            repository=workspace.relative_to(args.workspace_root).as_posix(),
        )
        prepared.append(
            {
                **task,
                "workspace_metadata": {
                    "source_files": source["source_files"],
                    "source_bytes": source["source_bytes"],
                    "source_sha256": source["sha256"],
                    "source_manifest": manifest_path.name,
                    "target_audit": targets,
                    "snapshot": snapshot,
                    "tree_digest_seconds": digest_seconds,
                    "prepare_seconds": time.monotonic() - started,
                    "patch_sha256": hashlib.sha256(
                        (task.get("patch") or "").encode()
                    ).hexdigest(),
                },
            }
        )
        print(
            json.dumps(
                {
                    "event": "prepare_complete",
                    "dataset_index": index,
                    "instance_id": task["instance_id"],
                    "seconds": prepared[-1]["workspace_metadata"]["prepare_seconds"],
                }
            ),
            flush=True,
        )

    task_path = args.output / "tasks.jsonl"
    task_path.write_text(
        "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in prepared),
        encoding="utf-8",
    )
    manifest = {
        "name": spec["name"],
        "seed": spec["seed"],
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": dataset_digest,
        "spec_sha256": hashlib.sha256(args.spec.read_bytes()).hexdigest(),
        "workspace_root": str(args.workspace_root.resolve()),
        "effective_task": spec["effective_task"],
        "tasks": [
            {
                key: task[key]
                for key in (
                    "dataset_index",
                    "instance_id",
                    "repo",
                    "base_commit",
                    "benchmark_split",
                    "difficulty",
                    "repository",
                    "workspace_metadata",
                )
            }
            for task in prepared
        ],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"tasks": len(prepared), "output": str(args.output)}))


if __name__ == "__main__":
    main()
