"""Fast, cached checkout creation for rollout workspaces."""

from __future__ import annotations

import fcntl
import shutil
import subprocess
import uuid
from pathlib import Path

CACHE_ROOT = Path("/tmp/codepin-repos")


def run_git(*args: str, input_text: str | None = None) -> None:
    subprocess.run(
        ["git", *args],
        input=input_text,
        check=True,
        capture_output=True,
        text=True,
    )


def ensure_mirror(repo_name: str) -> Path:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    name = repo_name.replace("/", "__")
    mirror = CACHE_ROOT / f"{name}.git"
    lock_path = CACHE_ROOT / f"{name}.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not mirror.is_dir():
            temporary = CACHE_ROOT / f"{name}.{uuid.uuid4().hex}.tmp"
            try:
                run_git(
                    "clone",
                    "--mirror",
                    f"https://github.com/{repo_name}.git",
                    str(temporary),
                )
                temporary.replace(mirror)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
    return mirror


def clone_instance(
    repo_name: str,
    commit_id: str | None,
    instance_id: str,
    output_dir: Path,
    patch: str | None = None,
) -> tuple[bool, Path | None]:
    """Create a disposable shared clone and optionally apply a mutation patch."""

    instance_path = output_dir / f"{repo_name.replace('/', '_')}_{instance_id}"
    try:
        mirror = ensure_mirror(repo_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        run_git("clone", "--shared", str(mirror), str(instance_path))
        if commit_id:
            run_git("-C", str(instance_path), "checkout", commit_id)
        if patch:
            run_git("-C", str(instance_path), "apply", input_text=patch)
        return True, instance_path
    except (OSError, subprocess.CalledProcessError):
        if instance_path.exists():
            shutil.rmtree(instance_path)
        return False, None
