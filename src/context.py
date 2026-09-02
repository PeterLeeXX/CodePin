"""Resolve real Python symbols and assemble a strict context character budget."""

from __future__ import annotations

from pathlib import Path

from src.build_swe_smith_code_search import SourceParseError, analyze_source
from src.tools.read_file import MAX_FILE_SIZE_BYTES, _resolve_file


def location_span(root: Path, location: dict) -> tuple[int, int]:
    path = _resolve_file(root.resolve(), location["file"])
    if not path.is_file():
        raise ValueError(f"location file does not exist: {location['file']}")
    if path.stat().st_size > MAX_FILE_SIZE_BYTES:
        raise ValueError("location exceeds the source size limit")
    text = path.read_text(encoding="utf-8")
    class_name, function = location.get("class_name"), location.get("function_name")
    if not class_name and not function:
        return 1, max(1, len(text.splitlines()))
    if path.suffix != ".py":
        raise ValueError("symbol localization currently supports Python files only")
    # Use the same AST symbol identities as the dataset builder.
    try:
        analysis = analyze_source(text, location["file"])
    except SourceParseError as exc:
        raise ValueError(str(exc)) from exc
    candidates = []
    if class_name:
        for cls, methods in analysis.classes:
            if cls.module == class_name:
                candidates.extend(methods if function else [cls])
    else:
        candidates = analysis.functions
    target = f"{class_name + '.' if class_name else ''}{function}"
    for candidate in candidates:
        if not function or candidate.entity == target:
            return candidate.start, candidate.end
    # Nested definitions are deliberately not guessed when the builder does not
    # represent them; callers receive a concrete validation error.
    raise ValueError(f"symbol does not exist: {location}")


def bounded_context(
    root: Path, locations: list[dict], max_chars: int = 12000, max_lines: int = 160
) -> list[dict]:
    if not 1 <= max_chars <= 30000 or not 1 <= max_lines <= 500:
        raise ValueError("context budget must be 1..30000 chars and 1..500 lines")
    snippets = []
    used_lines: set[tuple[str, int]] = set()
    remaining = max_chars
    line_budget = max_lines
    for location in locations:
        start, end = location_span(root, location)
        lines = (
            _resolve_file(root.resolve(), location["file"])
            .read_text(encoding="utf-8")
            .splitlines()
        )
        selected = []
        numbers = []
        for number in range(start, min(end, len(lines)) + 1):
            key = (location["file"], number)
            if key in used_lines:
                continue
            text = f"{number}: {lines[number - 1]}\n"
            if len(text) > remaining or line_budget == 0:
                break
            selected.append(text)
            numbers.append(number)
            used_lines.add(key)
            remaining -= len(text)
            line_budget -= 1
        snippets.append(
            {
                "file": location["file"],
                "symbol_start": start,
                "symbol_end": end,
                "line_numbers": numbers,
                "text": "".join(selected),
                "truncated": any(
                    (location["file"], n) not in used_lines
                    for n in range(start, end + 1)
                ),
            }
        )
    return snippets
