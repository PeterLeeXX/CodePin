from types import SimpleNamespace

import pytest

from src.tools.localization_finish import (
    CodeLocation,
    LocalizationFinishAction,
    LocalizationFinishExecutor,
    LocalizationFinishTool,
)


def _state(path):
    return SimpleNamespace(workspace=SimpleNamespace(working_dir=str(path)))


def test_localization_finish_schema_is_stable_and_read_only(tmp_path):
    tool = LocalizationFinishTool.create(_state(tmp_path))[0]
    schema = tool.to_openai_tool().function.parameters

    assert tool.name == "localization_finish"
    assert set(schema["properties"]) == {"locations"}
    assert schema["required"] == ["locations"]

    location_schema = schema["properties"]["locations"]["items"]
    assert set(location_schema["properties"]) == {
        "file",
        "class_name",
        "function_name",
    }
    assert location_schema["required"] == ["file"]
    assert tool.annotations.readOnlyHint is True
    assert tool.annotations.destructiveHint is False


def test_localization_finish_rejects_unsafe_and_duplicate_locations():
    with pytest.raises(ValueError):
        CodeLocation(file="../outside.py")
    with pytest.raises(ValueError):
        LocalizationFinishAction(
            locations=[CodeLocation(file="src/a.py"), CodeLocation(file="src/a.py")]
        )


def test_localization_finish_validates_files_before_finishing(tmp_path):
    executor = LocalizationFinishExecutor(str(tmp_path))
    action = LocalizationFinishAction(locations=[CodeLocation(file="missing.py")])

    observation = executor(action)

    assert observation.is_error
    assert "does not exist" in observation.text


def test_localization_finish_ends_conversation_after_valid_submission(tmp_path):
    (tmp_path / "module.py").write_text("value = 1\n", encoding="utf-8")
    conversation = SimpleNamespace(state=SimpleNamespace(execution_status="running"))
    executor = LocalizationFinishExecutor(str(tmp_path))
    action = LocalizationFinishAction(locations=[CodeLocation(file="module.py")])

    observation = executor(action, conversation)

    assert not observation.is_error
    assert conversation.state.execution_status.value == "finished"
    assert '"file": "module.py"' in observation.text
