from types import SimpleNamespace

from src.tools.read_file import ReadFileAction, ReadFileExecutor, ReadFileTool


def _state(path):
    return SimpleNamespace(workspace=SimpleNamespace(working_dir=str(path)))


def test_read_file_returns_requested_inclusive_range(tmp_path):
    (tmp_path / "module.py").write_text(
        "first\nsecond\nthird\nfourth\n",
        encoding="utf-8",
    )

    observation = ReadFileExecutor(str(tmp_path))(
        ReadFileAction(path="module.py", start_line=2, end_line=3)
    )

    assert not observation.is_error
    assert observation.path == "module.py"
    assert observation.start_line == 2
    assert observation.end_line == 3
    assert observation.total_lines == 4
    assert "     2→second" in observation.text
    assert "     3→third" in observation.text
    assert "first" not in observation.text
    assert not observation.truncated


def test_read_file_default_range_reports_continuation(tmp_path):
    (tmp_path / "module.py").write_text("line\n" * 201, encoding="utf-8")

    observation = ReadFileExecutor(str(tmp_path))(ReadFileAction(path="module.py"))

    assert not observation.is_error
    assert observation.end_line == 200
    assert observation.truncated
    assert "Use start_line=201" in observation.text


def test_read_file_caps_very_long_lines(tmp_path):
    (tmp_path / "minified.js").write_text("x" * 30_001, encoding="utf-8")

    observation = ReadFileExecutor(str(tmp_path))(
        ReadFileAction(path="minified.js", start_line=1, end_line=1)
    )

    assert not observation.is_error
    assert observation.truncated
    assert "output character limit" in observation.text


def test_read_file_handles_empty_file(tmp_path):
    (tmp_path / "empty.py").write_text("", encoding="utf-8")

    observation = ReadFileExecutor(str(tmp_path))(ReadFileAction(path="empty.py"))

    assert not observation.is_error
    assert observation.total_lines == 0
    assert observation.start_line == 1
    assert observation.end_line == 0
    assert observation.text == "File: empty.py is empty."


def test_read_file_rejects_binary_and_outside_paths(tmp_path):
    (tmp_path / "binary.dat").write_bytes(b"text\x00binary")
    executor = ReadFileExecutor(str(tmp_path))

    binary = executor(ReadFileAction(path="binary.dat"))
    outside = executor(ReadFileAction(path="../outside.py"))

    assert binary.is_error
    assert outside.is_error


def test_read_file_rejects_oversized_ranges(tmp_path):
    (tmp_path / "large.py").write_text("line\n" * 600, encoding="utf-8")

    observation = ReadFileExecutor(str(tmp_path))(
        ReadFileAction(path="large.py", start_line=1, end_line=501)
    )

    assert observation.is_error
    assert "at most 500 lines" in observation.text


def test_read_file_tool_schema_is_minimal_and_read_only(tmp_path):
    tool = ReadFileTool.create(_state(tmp_path))[0]
    schema = tool.to_openai_tool()["function"]["parameters"]

    assert tool.name == "read_file"
    assert set(schema["properties"]) == {"path", "start_line", "end_line"}
    assert schema["required"] == ["path"]
    assert tool.annotations.readOnlyHint is True
    assert tool.annotations.destructiveHint is False
