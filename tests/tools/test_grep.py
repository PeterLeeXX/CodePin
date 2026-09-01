from types import SimpleNamespace

import pytest

import src.tools.grep as grep_module
from src.tools.grep import GrepAction, GrepExecutor, GrepTool


def _state(path):
    return SimpleNamespace(workspace=SimpleNamespace(working_dir=str(path)))


def test_grep_returns_relative_lines(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "parser.py").write_text(
        "class Parser:\n    def parse(self):\n        return 1\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "parser.txt").write_text(
        "class ParserText:\n",
        encoding="utf-8",
    )

    observation = GrepExecutor(str(tmp_path))(
        GrepAction(pattern=r"class\s+Parser", path="src", include="*.py")
    )

    assert not observation.is_error
    assert observation.matches == ["src/parser.py:1:class Parser:"]
    assert observation.text == "src/parser.py:1:class Parser:"


def test_grep_supports_ripgrep_brace_globs(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "parser.ts").write_text(
        "class Parser:\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "parser.py").write_text(
        "class PythonParser:\n",
        encoding="utf-8",
    )

    observation = GrepExecutor(str(tmp_path))(
        GrepAction(pattern=r"class\s+Parser", path="src", include="*.{ts,tsx}")
    )

    assert not observation.is_error
    assert observation.matches == ["src/parser.ts:1:class Parser:"]


def test_grep_ripgrep_backend_caps_broad_searches(tmp_path):
    (tmp_path / "module.py").write_text("match\n" * 101, encoding="utf-8")

    observation = GrepExecutor(str(tmp_path))(GrepAction(pattern="match"))

    assert not observation.is_error
    assert observation.truncated
    assert len(observation.matches) == 100


def test_grep_no_match_is_not_an_error(tmp_path):
    (tmp_path / "module.py").write_text("value = 1\n", encoding="utf-8")

    observation = GrepExecutor(str(tmp_path))(GrepAction(pattern="missing_symbol"))

    assert not observation.is_error
    assert observation.matches == []
    assert observation.text == "No matches found."


def test_grep_rejects_invalid_regex_and_outside_path(tmp_path):
    executor = GrepExecutor(str(tmp_path))

    invalid_regex = executor(GrepAction(pattern="["))
    outside_path = executor(GrepAction(pattern="value", path="../"))

    assert invalid_regex.is_error
    assert outside_path.is_error


def test_grep_skips_binary_files(tmp_path):
    (tmp_path / "binary.dat").write_bytes(b"match\x00match")

    observation = GrepExecutor(str(tmp_path))(GrepAction(pattern="match"))

    assert not observation.is_error
    assert observation.matches == []


def test_grep_requires_ripgrep(tmp_path, monkeypatch):
    monkeypatch.setattr(grep_module.shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError, match="ripgrep"):
        GrepTool.create(_state(tmp_path))


def test_grep_tool_schema_is_minimal_and_read_only(tmp_path):
    tool = GrepTool.create(_state(tmp_path))[0]
    schema = tool.to_openai_tool()["function"]["parameters"]

    assert tool.name == "grep"
    assert set(schema["properties"]) == {"pattern", "path", "include"}
    assert schema["required"] == ["pattern"]
    assert tool.annotations.readOnlyHint is True
    assert tool.annotations.destructiveHint is False
