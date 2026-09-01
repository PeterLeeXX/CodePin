from types import SimpleNamespace

from src.tools.glob import GlobAction, GlobExecutor, GlobTool


def _state(path):
    return SimpleNamespace(workspace=SimpleNamespace(working_dir=str(path)))


def test_glob_returns_stable_relative_paths_and_filters_cache(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "z.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "hidden.py").write_text("", encoding="utf-8")

    observation = GlobExecutor(str(tmp_path))(GlobAction(pattern="**/*.py"))

    assert not observation.is_error
    assert observation.files == ["src/a.py", "src/z.py"]
    assert observation.search_path == "."


def test_glob_rejects_paths_and_patterns_outside_workspace(tmp_path):
    executor = GlobExecutor(str(tmp_path))

    absolute = executor(GlobAction(pattern="/etc/passwd"))
    traversal = executor(GlobAction(pattern="../*.py"))
    outside_path = executor(GlobAction(pattern="*.py", path="../"))

    assert absolute.is_error
    assert traversal.is_error
    assert outside_path.is_error


def test_glob_does_not_return_symlinks_resolving_outside_workspace(tmp_path):
    outside = tmp_path.parent / "outside-codepin-glob.py"
    outside.write_text("secret", encoding="utf-8")
    (tmp_path / "linked.py").symlink_to(outside)

    observation = GlobExecutor(str(tmp_path))(GlobAction(pattern="*.py"))

    assert not observation.is_error
    assert observation.files == []


def test_glob_caps_and_sorts_results(tmp_path):
    for index in reversed(range(250)):
        (tmp_path / f"file_{index:03}.py").write_text("", encoding="utf-8")

    observation = GlobExecutor(str(tmp_path))(GlobAction(pattern="*.py"))

    assert observation.truncated
    assert len(observation.files) == 100
    assert observation.files == [f"file_{index:03}.py" for index in range(100)]


def test_glob_tool_schema_is_minimal_and_read_only(tmp_path):
    tool = GlobTool.create(_state(tmp_path))[0]
    schema = tool.to_openai_tool()["function"]["parameters"]

    assert tool.name == "glob"
    assert set(schema["properties"]) == {"pattern", "path"}
    assert schema["required"] == ["pattern"]
    assert tool.annotations.readOnlyHint is True
    assert tool.annotations.destructiveHint is False
