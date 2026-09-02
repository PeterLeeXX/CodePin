"""Structured file, module, and function localization reward."""

from __future__ import annotations

from typing import Any


def f1(predicted: set[str], truth: set[str]) -> float:
    if not truth:
        return 0.0
    true_positive = len(predicted & truth)
    if not true_positive:
        return 0.0
    precision = true_positive / len(predicted)
    recall = true_positive / len(truth)
    return 2 * precision * recall / (precision + recall)


def parse_locations(
    locations: list[dict[str, Any]],
) -> tuple[set[str], set[str], set[str]]:
    files: set[str] = set()
    modules: set[str] = set()
    entities: set[str] = set()
    for location in locations:
        path = location.get("file")
        class_name = location.get("class_name")
        function_name = location.get("function_name")
        if not path:
            return set(), set(), set()
        files.add(path)
        if class_name:
            modules.add(f"{path}:{class_name}")
        elif function_name:
            modules.add(f"{path}:{function_name}")
        if class_name and function_name:
            entities.add(f"{path}:{class_name}.{function_name}")
        elif function_name:
            entities.add(f"{path}:{function_name}")
    return files, modules, entities


def multilevel_localization_f1_reward(
    *,
    instance: dict[str, Any],
    structured_locations: list[dict[str, Any]] | None,
    **_: Any,
) -> tuple[float, dict[str, float]]:
    if not structured_locations:
        scores = {
            "file_reward": 0.0,
            "module_reward": 0.0,
            "entity_reward": 0.0,
            "file_f1": 0.0,
            "class_f1": 0.0,
            "function_f1": 0.0,
        }
        return 0.0, {"multilevel_localization_f1_reward": 0.0, **scores}

    truth_files: set[str] = set()
    truth_modules: set[str] = set()
    truth_entities: set[str] = set()
    for change in instance.get("file_changes", []):
        if path := change.get("file"):
            truth_files.add(path)
        details = change.get("changes") or {}
        for key in ("edited_modules", "added_modules"):
            truth_modules.update(details.get(key) or [])
        for key in ("edited_entities", "added_entities"):
            truth_entities.update(details.get(key) or [])

    predicted = parse_locations(structured_locations)
    scores = {
        "file_reward": f1(predicted[0], truth_files),
        "module_reward": f1(predicted[1], truth_modules),
        "entity_reward": f1(predicted[2], truth_entities),
    }
    total = sum(scores.values())
    truth_classes = truth_modules - truth_entities
    predicted_classes = {
        f"{location['file']}:{location['class_name']}"
        for location in structured_locations
        if location.get("class_name")
    }
    scores.update(
        file_f1=scores["file_reward"],
        class_f1=f1(predicted_classes, truth_classes),
        function_f1=scores["entity_reward"],
    )
    return total, {"multilevel_localization_f1_reward": total, **scores}
