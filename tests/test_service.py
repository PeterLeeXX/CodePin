import asyncio
import os
import time

import pytest

from src.context import bounded_context, location_span
from src.service import (
    LocalizationRequest,
    LocalizationService,
    ResultCache,
    ServiceConfig,
    tree_digest,
)
from src.tools.localization_finish import (
    LocalizationFinishAction,
    LocalizationFinishExecutor,
)


def test_context_resolves_symbols_and_obeys_total_budget(tmp_path):
    (tmp_path / "app.py").write_text(
        "class A:\n    def run(self):\n        return 1\n\ndef f():\n    return 2\n"
    )
    locations = [
        {"file": "app.py", "class_name": "A", "function_name": "run"},
        {"file": "app.py", "class_name": "A"},
    ]
    assert location_span(tmp_path, locations[0]) == (2, 3)
    snippets = bounded_context(tmp_path, locations, max_chars=55, max_lines=2)
    assert sum(len(s["text"]) for s in snippets) <= 55
    assert sum(len(s["line_numbers"]) for s in snippets) <= 2
    numbers = [n for s in snippets for n in s["line_numbers"]]
    assert len(numbers) == len(set(numbers))
    assert any(s["truncated"] for s in snippets)
    assert location_span(tmp_path, {"file": "app.py", "function_name": "f"}) == (5, 6)


def test_finish_rejects_invented_symbols(tmp_path):
    (tmp_path / "app.py").write_text("def present():\n    return 1\n")
    action = LocalizationFinishAction(
        locations=[{"file": "app.py", "function_name": "absent"}]
    )
    result = LocalizationFinishExecutor(str(tmp_path))(action)
    assert result.is_error
    assert "symbol does not exist" in result.text


def test_digest_invalidates_same_size_edits_untracked_and_links(tmp_path):
    file = tmp_path / "app.py"
    file.write_text("old")
    before = tree_digest(tmp_path)
    stamp = file.stat()
    file.write_text("new")
    os.utime(file, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert tree_digest(tmp_path) != before
    before = tree_digest(tmp_path)
    (tmp_path / "untracked.txt").write_text("data")
    assert tree_digest(tmp_path) != before
    (tmp_path / "outside").symlink_to(tmp_path.parent)
    with pytest.raises(ValueError):
        location_span(tmp_path, {"file": "outside/anything.py"})


def test_cache_lru_ttl_copy_and_failed_results():
    cache = ResultCache(1, 0.01)
    result = {"status": "ok", "locations": ["a"]}
    cache.put("a", result)
    cache.get("a")["locations"].append("b")
    assert cache.get("a")["locations"] == ["a"]
    cache.put("b", result)
    assert cache.get("a") is None
    cache.put("failure", {"status": "error"})
    assert cache.get("failure") is None
    time.sleep(0.02)
    assert cache.get("b") is None


def test_cache_key_separates_deployments_issues_and_budgets(tmp_path):
    deployment = tmp_path / "deployment.json"
    deployment.write_text('{"id": "one"}')
    config = ServiceConfig(tmp_path, deployment_file=deployment)
    service = LocalizationService(config)
    request = LocalizationRequest(repository=".", issue="find f")
    original = service.cache_key(request, "snapshot")
    assert (
        service.cache_key(request.model_copy(update={"issue": "find g"}), "snapshot")
        != original
    )
    assert (
        service.cache_key(
            request.model_copy(update={"max_context_chars": 20}), "snapshot"
        )
        != original
    )
    assert service.cache_key(request, "changed_snapshot") != original
    deployment.write_text('{"id": "two"}')
    assert service.cache_key(request, "snapshot") != original
    with pytest.raises(ValueError, match="disable result caching"):
        asyncio.run(service.localize(request, purpose="rollout"))
    with pytest.raises(ValueError):
        service.repository("../")


def test_cache_requires_deployment_identity(tmp_path):
    with pytest.raises(ValueError, match="deployment"):
        ServiceConfig(tmp_path)
    assert ServiceConfig(tmp_path, cache_size=0)


def test_digest_includes_empty_and_ignored_paths_without_following_links(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    empty = tree_digest(repo)
    (repo / ".git").mkdir()
    assert tree_digest(repo) != empty
    (repo / ".git" / "ignored").write_text("one")
    original = tree_digest(repo)
    (repo / ".git" / "ignored").write_text("two")
    assert tree_digest(repo) != original
    external = tmp_path / "external"
    external.write_text("outside")
    (repo / "link").symlink_to(external)
    original = tree_digest(repo)
    external.write_text("changed outside")
    assert tree_digest(repo) == original
    (repo / "link").unlink()
    (repo / "link").symlink_to(repo / ".git" / "ignored")
    assert tree_digest(repo) != original


def test_uncached_service_reports_stage_timings(tmp_path, monkeypatch):
    (tmp_path / "app.py").write_text("def target():\n    return 1\n")

    def localize(*_args, **_kwargs):
        return {
            "status": "ok",
            "structured_locations": [{"file": "app.py", "function_name": "target"}],
            "metrics": {},
            "errors": [],
        }

    monkeypatch.setattr("src.service.run_localization", localize)
    service = LocalizationService(ServiceConfig(tmp_path, cache_size=0))
    result = asyncio.run(
        service.localize(LocalizationRequest(repository=".", issue="find target"))
    )

    assert result["status"] == "ok"
    for key in (
        "service_queue_seconds",
        "service_total_seconds",
        "repository_digest_before_seconds",
        "cache_key_before_seconds",
        "rollout_seconds",
        "bounded_context_seconds",
        "repository_digest_after_seconds",
        "cache_key_after_seconds",
    ):
        assert result["metrics"][key] >= 0
