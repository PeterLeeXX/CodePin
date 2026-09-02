"""Shared trajectory validation and observable tool costs for serving and SkyRL."""

from __future__ import annotations

import json
import posixpath
from typing import Any

from src.rewards.file_localization.file_localization import (
    multilevel_localization_f1_reward,
)

TOOL_NAMES = {"glob", "grep", "read_file", "localization_finish"}


def observation_text(observation: dict) -> str:
    if isinstance(observation.get("text"), str):
        return observation["text"]
    return "\n".join(
        item.get("text", "")
        for item in (observation.get("content") or [])
        if isinstance(item, dict) and isinstance(item.get("text"), str)
    )


def validate_events(events: list[dict]) -> tuple[list[dict] | None, list[str]]:
    """Require executed, paired tools and one successful, sole final finish."""
    from src.tools.glob import GlobAction
    from src.tools.grep import GrepAction
    from src.tools.localization_finish import LocalizationFinishAction
    from src.tools.read_file import ReadFileAction

    schemas = dict(
        zip(
            ("glob", "grep", "read_file", "localization_finish"),
            (GlobAction, GrepAction, ReadFileAction, LocalizationFinishAction),
            strict=True,
        )
    )
    if not isinstance(events, list) or any(not isinstance(e, dict) for e in events):
        return None, ["invalid_events"]
    if any(
        e.get("kind") == "ObservationEvent"
        and not isinstance(e.get("observation"), dict)
        for e in events
    ):
        return None, ["invalid_observation"]
    errors: list[str] = []
    actions = [e for e in events if e.get("kind") == "ActionEvent"]
    observations = [e for e in events if e.get("kind") == "ObservationEvent"]
    ids = [e.get("tool_call_id") for e in actions]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        return None, ["invalid_tool_call_ids"]
    for event in events:
        if event.get("kind") in {"AgentErrorEvent", "ConversationErrorEvent"}:
            errors.append("agent_error")
    for action in actions:
        if action.get("tool_name") not in TOOL_NAMES or not isinstance(
            action.get("action"), dict
        ):
            errors.append("invalid_action")
        else:
            try:
                schemas[action["tool_name"]].model_validate(action["action"])
            except ValueError:
                errors.append("invalid_action")
        matches = [
            o
            for o in observations
            if o.get("tool_call_id") == action.get("tool_call_id")
        ]
        if len(matches) != 1:
            errors.append("unpaired_action")
        elif matches[0]["observation"].get("is_error"):
            errors.append("tool_error")
        elif matches[0].get("tool_name") != action.get("tool_name") or events.index(
            matches[0]
        ) < events.index(action):
            errors.append("observation_mismatch")
    if any(o.get("tool_call_id") not in ids for o in observations):
        errors.append("orphan_observation")
    finishes = [a for a in actions if a.get("tool_name") == "localization_finish"]
    locations = None
    if len(finishes) != 1:
        errors.append("missing_or_multiple_finish")
    else:
        finish = finishes[0]
        siblings = [
            a
            for a in actions
            if a.get("llm_response_id") == finish.get("llm_response_id")
        ]
        if actions[-1] is not finish or len(siblings) != 1:
            errors.append("finish_not_sole_final_call")
        else:
            action = finish.get("action")
            locations = action.get("locations") if isinstance(action, dict) else None
            if not locations:
                errors.append("empty_finish")
    return (None if errors else locations), sorted(set(errors))


def tool_metrics(events: list[dict]) -> dict[str, float]:
    """Count cost, never give novelty credit for unrelated content."""
    metrics = dict.fromkeys(
        (
            "num_tool_calls",
            "search_calls",
            "read_calls",
            "repeated_searches",
            "read_lines",
            "overlap_lines",
            "output_chars",
            "excess_output_chars",
            "truncated_outputs",
            "tool_errors",
            "completion_tokens",
            "num_turns",
        ),
        0.0,
    )
    seen_searches: set[str] = set()
    seen_lines: dict[str, set[int]] = {}
    for event in events if isinstance(events, list) else []:
        if not isinstance(event, dict):
            continue
        kind = event.get("kind")
        name = event.get("tool_name")
        if kind == "TokenEvent":
            metrics["num_turns"] += 1
            ids = event.get("response_token_ids")
            metrics["completion_tokens"] += len(ids) if isinstance(ids, list) else 0
        elif kind == "ActionEvent":
            metrics["num_tool_calls"] += 1
            action = event.get("action")
            if not isinstance(action, dict):
                continue
            if name in {"glob", "grep"}:
                metrics["search_calls"] += 1
                args = {
                    "pattern": action.get("pattern"),
                    "path": posixpath.normpath(action.get("path") or "."),
                }
                if name == "grep":
                    args["include"] = action.get("include") or None
                key = json.dumps([name, args], sort_keys=True)
                metrics["repeated_searches"] += key in seen_searches
                seen_searches.add(key)
            elif name == "read_file":
                metrics["read_calls"] += 1
        elif kind == "ObservationEvent":
            obs = event.get("observation")
            if not isinstance(obs, dict):
                metrics["tool_errors"] += 1
                continue
            metrics["tool_errors"] += bool(obs.get("is_error"))
            metrics["truncated_outputs"] += bool(obs.get("truncated"))
            if name == "localization_finish":
                continue
            size = len(observation_text(obs))
            metrics["output_chars"] += size
            metrics["excess_output_chars"] += max(0, size - 8000)
            if name == "read_file" and not obs.get("is_error"):
                if (
                    not isinstance(obs.get("path"), str)
                    or type(obs.get("start_line")) is not int
                    or type(obs.get("end_line")) is not int
                    or not 0 <= obs["end_line"] - obs["start_line"] + 1 <= 500
                ):
                    metrics["tool_errors"] += 1
                    continue
                path = posixpath.normpath(obs["path"])
                lines = set(range(obs["start_line"], obs["end_line"] + 1))
                previous = seen_lines.setdefault(path, set())
                metrics["read_lines"] += len(lines)
                metrics["overlap_lines"] += len(lines & previous)
                previous.update(lines)
    reads = max(1, metrics["read_lines"])
    metrics["overlap_ratio"] = metrics["overlap_lines"] / reads
    metrics["tool_efficiency_cost"] = (
        0.01 * max(0, metrics["num_tool_calls"] - 1)
        + 0.10 * metrics["repeated_searches"]
        + 0.20 * metrics["overlap_ratio"]
        + metrics["output_chars"] / 100_000
        + metrics["excess_output_chars"] / 50_000
        + 0.05 * metrics["truncated_outputs"]
    )
    return metrics


def score_trajectory(
    instance: dict[str, Any],
    locations: list[dict] | None,
    events: list[dict],
    efficiency_weight: float = 0.2,
    valid: bool = True,
) -> tuple[float, dict, dict]:
    if not 0 <= efficiency_weight <= 1:
        raise ValueError("efficiency_weight must be in [0, 1]")
    quality, details = multilevel_localization_f1_reward(
        instance=instance,
        structured_locations=locations if valid else None,
    )
    metrics = tool_metrics(events)
    penalty = min(quality, efficiency_weight * metrics["tool_efficiency_cost"])
    total = quality - penalty
    details.update(
        tool_efficiency_penalty=penalty,
        total_reward=total,
        trajectory_valid=float(valid),
    )
    return total, details, metrics
