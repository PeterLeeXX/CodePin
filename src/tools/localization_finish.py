"""Structured completion tool for CodePin localization tasks."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator
from rich.text import Text

from src.profiling import nvtx_range

if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
    from openhands.sdk.conversation.state import ConversationState


class CodeLocation(BaseModel):
    """One existing repository location selected for modification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    file: str = Field(description="Repository-relative path to an existing file")
    class_name: str | None = Field(default=None, description="Optional class name")
    function_name: str | None = Field(
        default=None,
        description="Optional function or method name",
    )

    @field_validator("file")
    @classmethod
    def validate_file(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        parsed = PurePosixPath(normalized)
        if not normalized or normalized != normalized.strip():
            raise ValueError("file must be a non-empty repository-relative path")
        if (
            parsed.is_absolute()
            or normalized.startswith("./")
            or ".." in parsed.parts
            or re.match(r"^[A-Za-z]:/", normalized)
        ):
            raise ValueError(
                "file must be relative without a leading './' or '..' component"
            )
        return parsed.as_posix()

    @field_validator("class_name", "function_name")
    @classmethod
    def validate_optional_name(cls, value: str | None) -> str | None:
        if value is not None and (not value or value != value.strip()):
            raise ValueError("class and function names cannot be empty or padded")
        return value


class LocalizationFinishAction(Action):
    """Submit the final set of source locations."""

    locations: list[CodeLocation] = Field(
        min_length=1,
        max_length=64,
        description=(
            "Locations that require modification. Each item has a repository-relative "
            "file and optional class_name and function_name."
        ),
    )

    @field_validator("locations")
    @classmethod
    def reject_duplicates(cls, locations: list[CodeLocation]) -> list[CodeLocation]:
        signatures = {
            (location.file, location.class_name, location.function_name)
            for location in locations
        }
        if len(signatures) != len(locations):
            raise ValueError("locations cannot contain duplicate entries")
        return locations

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append("Submitting localization results:\n", style="bold blue")
        for index, location in enumerate(self.locations, start=1):
            content.append(f"  {index}. {location.file}", style="cyan")
            if location.class_name:
                content.append(f" → {location.class_name}", style="yellow")
            if location.function_name:
                content.append(f".{location.function_name}", style="magenta")
            content.append("\n")
        return content


class LocalizationFinishObservation(Observation):
    """Confirmation returned as the conversation transitions to finished."""

    @property
    def visualize(self) -> Text:
        return Text()


class LocalizationFinishExecutor(
    ToolExecutor[LocalizationFinishAction, LocalizationFinishObservation]
):
    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root).resolve()

    @nvtx_range("codepin.tool.localization_finish")
    def __call__(
        self,
        action: LocalizationFinishAction,
        conversation: LocalConversation | None = None,
    ) -> LocalizationFinishObservation:
        try:
            from src.context import location_span

            for location in action.locations:
                candidate = (self.workspace_root / location.file).resolve()
                candidate.relative_to(self.workspace_root)
                if not candidate.is_file():
                    raise ValueError(f"location file does not exist: {location.file}")
                location_span(self.workspace_root, location.model_dump())
            if conversation is None:
                raise ValueError("localization_finish requires an active conversation")

            payload = [location.model_dump() for location in action.locations]
            conversation.state.execution_status = ConversationExecutionStatus.FINISHED
            return LocalizationFinishObservation.from_text(
                text=json.dumps(payload, ensure_ascii=False, indent=2)
            )
        except (AttributeError, OSError, ValueError) as exc:
            return LocalizationFinishObservation.from_text(
                text=str(exc),
                is_error=True,
            )


TOOL_DESCRIPTION = """Submit the final CodePin localization result and end the run.

Each location must name an existing repository-relative file. Add class_name and
function_name only when that existing symbol is the specific modification target.
Submit each distinct location once. This tool must be the only call in the final turn.
"""


class LocalizationFinishTool(
    ToolDefinition[LocalizationFinishAction, LocalizationFinishObservation]
):
    """OpenHands definition for CodePin's structured final submission."""

    @classmethod
    def create(
        cls,
        conv_state: ConversationState,
        **params,
    ) -> Sequence[LocalizationFinishTool]:
        if params:
            raise ValueError("LocalizationFinishTool does not accept parameters")
        working_dir = conv_state.workspace.working_dir
        if not Path(working_dir).is_dir():
            raise ValueError(f"working_dir {working_dir!r} is not a valid directory")
        return [
            cls(
                action_type=LocalizationFinishAction,
                observation_type=LocalizationFinishObservation,
                description=TOOL_DESCRIPTION,
                executor=LocalizationFinishExecutor(working_dir),
                annotations=ToolAnnotations(
                    title="localization_finish",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            )
        ]


register_tool(LocalizationFinishTool.name, LocalizationFinishTool)
