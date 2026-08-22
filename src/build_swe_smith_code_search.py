"""Build the code-localization version of SWE-Smith from the raw parquet shards.

The raw SWE-Smith patches introduce bugs into clean repository snapshots.  This
pipeline downloads the clean Python files from the corresponding ``swesmith``
GitHub snapshot repositories, applies each patch in memory, and maps changed
lines to file, class/module, and function/method targets.

The implementation is based on LocAgent's ``gen_oracle_locations.py`` but is
adapted for SWE-Smith and fixes the pieces needed for an end-to-end dataset
build: sharded parquet input, multi-file patches, source caching, deterministic
train/validation splitting, structured parquet output, and a cleaning report.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import random
import sys
import time
import tokenize
import warnings
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from unidiff import PatchSet
from unidiff.errors import UnidiffParseError


DEFAULT_INPUT = Path("data/orgin_SWE_smith")
DEFAULT_OUTPUT = Path("data/SWE-smith-code-search")
DEFAULT_CACHE = Path("data/.cache/swe_smith_sources")


class BuildError(Exception):
    """Base exception for a recoverable per-instance build failure."""


class PatchApplyError(BuildError):
    """Raised when a patch does not match the downloaded repository snapshot."""


class SourceFetchError(BuildError):
    """Raised when a source file cannot be downloaded or decoded."""


class SourceParseError(BuildError):
    """Raised when Python's AST parser cannot parse a source file."""


@dataclass(frozen=True)
class SourceKey:
    repo: str
    path: str


@dataclass(frozen=True)
class Location:
    """A code location containing the line and target names used by CodePin."""

    start: int
    end: int
    module: str | None
    entity: str | None
    identity: tuple[str, str]


@dataclass
class SourceAnalysis:
    classes: list[tuple[Location, list[Location]]]
    functions: list[Location]
    ignored_spans: list[tuple[int, int]]
    identities: set[tuple[str, str]]

    def location_at(self, line: int) -> Location | None:
        # Match LocAgent's lookup precedence exactly: classes are visited in
        # ``ast.walk`` order, direct methods are checked first, and a line in a
        # class that is not in one of those methods resolves to the class.  In
        # particular, an outer class intentionally shadows nested classes.
        for class_location, methods in self.classes:
            for method in methods:
                if method.start <= line <= method.end:
                    return method
            if class_location.start <= line <= class_location.end:
                return class_location
        for function in self.functions:
            if function.start <= line <= function.end:
                return function
        return None

    def ignores(self, line: int) -> bool:
        return any(start <= line <= end for start, end in self.ignored_spans)


CHANGE_TYPE = pa.struct(
    [
        pa.field("added_entities", pa.list_(pa.string())),
        pa.field("added_modules", pa.list_(pa.string())),
        pa.field("edited_entities", pa.list_(pa.string())),
        pa.field("edited_modules", pa.list_(pa.string())),
    ]
)
FILE_CHANGE_TYPE = pa.struct(
    [pa.field("changes", CHANGE_TYPE), pa.field("file", pa.string())]
)
PROMPT_TYPE = pa.struct(
    [pa.field("content", pa.string()), pa.field("role", pa.string())]
)
OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("instance_id", pa.string()),
        pa.field("file_changes", pa.list_(FILE_CHANGE_TYPE)),
        pa.field("repo", pa.string()),
        pa.field("base_commit", pa.null()),
        pa.field("problem_statement", pa.string()),
        pa.field("patch", pa.string()),
        pa.field("target", pa.list_(FILE_CHANGE_TYPE)),
        pa.field("prompt", pa.list_(PROMPT_TYPE)),
        pa.field("use_patch", pa.bool_()),
    ]
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean raw SWE-Smith parquet shards into code-search targets."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--fetch-retries", type=int, default=4)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only scan the first N raw rows (intended for smoke tests).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing output files."
    )
    parser.add_argument(
        "--keep-all",
        action="store_true",
        help="Keep the intermediate all.parquet alongside train/validation files.",
    )
    return parser.parse_args(argv)


def parquet_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        paths = [input_path]
    else:
        paths = sorted(input_path.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {input_path}")
    return paths


def iter_raw_rows(
    paths: Sequence[Path], columns: Sequence[str], batch_size: int, limit: int | None
) -> Iterator[dict]:
    seen = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=list(columns), batch_size=batch_size):
            for row in batch.to_pylist():
                if limit is not None and seen >= limit:
                    return
                seen += 1
                yield row


def parse_patch(patch: str) -> PatchSet:
    try:
        return PatchSet(patch or "")
    except (UnidiffParseError, UnicodeDecodeError, ValueError) as exc:
        raise BuildError(f"invalid_patch:{exc}") from exc


def eligible_python_files(row: dict) -> tuple[PatchSet | None, list]:
    if not (row.get("problem_statement") or "").strip():
        return None, []
    patch_set = parse_patch(row.get("patch") or "")
    if any(file.is_added_file or file.is_removed_file for file in patch_set):
        return None, []
    python_files = [file for file in patch_set if str(file.path).endswith(".py")]
    if not python_files:
        return None, []
    return patch_set, python_files


def source_cache_path(cache_dir: Path, key: SourceKey) -> Path:
    safe_parts = [part for part in Path(key.path).parts if part not in {"", ".", ".."}]
    return cache_dir / key.repo.replace("/", "__") / Path(*safe_parts)


def source_url(key: SourceKey) -> str:
    encoded_path = "/".join(urllib.parse.quote(part) for part in key.path.split("/"))
    return f"https://raw.githubusercontent.com/{key.repo}/HEAD/{encoded_path}"


def fetch_source(
    key: SourceKey, cache_dir: Path, retries: int, timeout: int = 45
) -> Path:
    destination = source_cache_path(cache_dir, key)
    if destination.is_file() and destination.stat().st_size:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        source_url(key), headers={"User-Agent": "CodePin-SWE-Smith-builder/1.0"}
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            if not payload:
                raise SourceFetchError("empty response")
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(payload)
            temporary.replace(destination)
            return destination
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    raise SourceFetchError(f"{source_url(key)}: {last_error}")


def decode_python_source(payload: bytes, label: str) -> str:
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(payload).readline)
        return payload.decode(encoding)
    except (SyntaxError, UnicodeDecodeError, LookupError) as exc:
        raise SourceFetchError(f"cannot decode {label}: {exc}") from exc


def same_patch_line(source_line: str, patch_line: str) -> bool:
    return source_line.rstrip("\r\n") == patch_line.rstrip("\r\n")


def apply_file_patch(source: str, patched_file) -> str:
    """Apply one unidiff PatchedFile to source text without touching a checkout."""

    source_lines = source.splitlines(keepends=True)
    result: list[str] = []
    cursor = 1

    for hunk in patched_file:
        if hunk.source_start < cursor:
            raise PatchApplyError(
                f"overlapping hunks in {patched_file.path} at {hunk.source_start}"
            )
        result.extend(source_lines[cursor - 1 : hunk.source_start - 1])
        cursor = hunk.source_start

        for line in hunk:
            # ``unidiff`` exposes "\\ No newline at end of file" as a
            # synthetic line.  It consumes neither source nor target text.
            if line.line_type == "\\":
                continue
            value = str(line)[1:]
            if line.is_added:
                result.append(value)
                continue
            if cursor > len(source_lines):
                raise PatchApplyError(
                    f"{patched_file.path}:{cursor}: patch extends past source"
                )
            actual = source_lines[cursor - 1]
            if not same_patch_line(actual, value):
                raise PatchApplyError(
                    f"{patched_file.path}:{cursor}: source does not match patch context"
                )
            if line.is_context:
                result.append(actual)
            cursor += 1

    result.extend(source_lines[cursor - 1 :])
    return "".join(result)


def node_start(node: ast.AST) -> int:
    # LocAgent starts spans at the ``def``/``class`` line.  Decorator-only
    # edits therefore resolve to the containing class or file, not the method.
    return getattr(node, "lineno", 1)


def add_docstring_span(node: ast.AST, spans: list[tuple[int, int]]) -> None:
    body = getattr(node, "body", None)
    if not body:
        return
    first = body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        spans.append((first.lineno, getattr(first, "end_lineno", first.lineno)))


def analyze_source(source: str, label: str) -> SourceAnalysis:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source, filename=label)
    except (SyntaxError, ValueError, TypeError) as exc:
        raise SourceParseError(f"cannot parse {label}: {exc}") from exc

    walked = list(ast.walk(tree))
    class_nodes = [node for node in walked if isinstance(node, ast.ClassDef)]
    class_method_names: set[str] = set()
    classes: list[tuple[Location, list[Location]]] = []
    functions: list[Location] = []
    identities: set[tuple[str, str]] = set()
    ignored_spans: list[tuple[int, int]] = []

    for node in class_nodes:
        class_identity = ("class", node.name)
        class_location = Location(
            node_start(node),
            getattr(node, "end_lineno", node.lineno),
            node.name,
            None,
            class_identity,
        )
        identities.add(class_identity)
        methods: list[Location] = []
        for child in node.body:
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            class_method_names.add(child.name)
            qualname = f"{node.name}.{child.name}"
            identity = ("function", qualname)
            identities.add(identity)
            methods.append(
                Location(
                    node_start(child),
                    getattr(child, "end_lineno", child.lineno),
                    node.name,
                    qualname,
                    identity,
                )
            )
        classes.append((class_location, methods))
        add_docstring_span(node, ignored_spans)

    # This name-based exclusion mirrors LocAgent's parser, including its
    # behavior for nested functions and same-named methods in different scopes.
    for node in walked:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name in class_method_names:
            continue
        identity = ("function", node.name)
        identities.add(identity)
        functions.append(
            Location(
                node_start(node),
                getattr(node, "end_lineno", node.lineno),
                node.name,
                node.name,
                identity,
            )
        )

    # The dataset curation ignores edits that only touch docstrings at either
    # class or function granularity.
    for node in walked:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add_docstring_span(node, ignored_spans)

    for node in walked:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            ignored_spans.append(
                (node.lineno, getattr(node, "end_lineno", node.lineno))
            )
    for statement in tree.body:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            ignored_spans.append(
                (statement.lineno, getattr(statement, "end_lineno", statement.lineno))
            )

    # LocAgent ignores standalone comment edits but still counts code lines
    # with trailing inline comments.
    source_lines = source.splitlines()
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type != tokenize.COMMENT:
                continue
            prefix = source_lines[token.start[0] - 1][: token.start[1]]
            if not prefix.strip():
                ignored_spans.append((token.start[0], token.end[0]))
    except (IndentationError, tokenize.TokenError):
        pass

    return SourceAnalysis(classes, functions, ignored_spans, identities)


def append_unique(values: list[str], value: str | None) -> None:
    if value and value not in values:
        values.append(value)


def qualify(path: str, name: str | None) -> str | None:
    return f"{path}:{name}" if name else None


def file_change_from_patch(
    path: str,
    original: str,
    patched: str,
    patched_file,
    old_analysis: SourceAnalysis | None = None,
) -> dict:
    old = old_analysis or analyze_source(original, f"{path} (original)")
    new = analyze_source(patched, f"{path} (patched)")
    changes: dict[str, list[str]] = {
        "added_entities": [],
        "added_modules": [],
        "edited_entities": [],
        "edited_modules": [],
    }

    def record(analysis: SourceAnalysis, line_number: int, removed_side: bool) -> None:
        if analysis.ignores(line_number):
            return
        location = analysis.location_at(line_number)
        if location is None:
            return

        # SWE-Smith's patch is a mutation that introduces the bug, whereas the
        # localization target describes the inverse (the eventual fix).  An
        # entity removed by the mutation is therefore an entity added by the
        # fix.  New buggy entities have no "deleted" output bucket and remain
        # edited targets, matching LocAgent's target schema.
        entity_prefix = "edited"
        if removed_side and location.entity and location.identity not in new.identities:
            entity_prefix = "added"

        module_identity = location.identity
        if location.entity and location.module != location.entity:
            module_identity = ("class", location.module or "")
        module_prefix = "edited"
        if removed_side and module_identity not in new.identities:
            module_prefix = "added"

        append_unique(
            changes[f"{module_prefix}_modules"], qualify(path, location.module)
        )
        append_unique(
            changes[f"{entity_prefix}_entities"], qualify(path, location.entity)
        )

    for hunk in patched_file:
        removed_lines = [line for line in hunk if line.is_removed]
        added_lines = [line for line in hunk if line.is_added]
        for line in removed_lines:
            if line.source_line_no is not None and line.value.strip():
                record(old, line.source_line_no, True)
        for line in added_lines:
            if line.target_line_no is not None and line.value.strip():
                record(new, line.target_line_no, False)

    nullable_changes = {key: value or None for key, value in changes.items()}
    return {"changes": nullable_changes, "file": path}


def process_instance(
    row: dict,
    cache_dir: Path,
    source_materials: dict[SourceKey, tuple[str, SourceAnalysis]],
) -> dict:
    patch_set, python_files = eligible_python_files(row)
    if patch_set is None:
        raise BuildError("instance was not eligible")

    file_changes: list[dict] = []
    for patched_file in python_files:
        path = str(patched_file.path)
        key = SourceKey(row["repo"], path)
        cache_path = source_cache_path(cache_dir, key)
        if not cache_path.is_file():
            raise SourceFetchError(f"missing cached source: {key.repo}/{key.path}")
        material = source_materials.get(key)
        if material is None:
            original = decode_python_source(cache_path.read_bytes(), str(cache_path))
            original_analysis = analyze_source(original, f"{path} (original)")
            material = (original, original_analysis)
            source_materials[key] = material
        original, original_analysis = material
        patched = apply_file_patch(original, patched_file)
        file_changes.append(
            file_change_from_patch(
                path, original, patched, patched_file, old_analysis=original_analysis
            )
        )

    prompt = [{"content": row["problem_statement"], "role": "user"}]
    return {
        "instance_id": row["instance_id"],
        "file_changes": file_changes,
        "repo": row["repo"],
        "base_commit": None,
        "problem_statement": row["problem_statement"],
        "patch": row["patch"],
        "target": file_changes,
        "prompt": prompt,
        "use_patch": True,
    }


def classify_ineligible(row: dict) -> tuple[str | None, list[SourceKey]]:
    if not (row.get("problem_statement") or "").strip():
        return "empty_problem_statement", []
    try:
        patch_set = parse_patch(row.get("patch") or "")
    except BuildError:
        return "invalid_patch", []
    if any(file.is_added_file or file.is_removed_file for file in patch_set):
        return "creates_or_deletes_file", []
    python_paths = [str(file.path) for file in patch_set if str(file.path).endswith(".py")]
    if not python_paths:
        return "no_python_changes", []
    return None, [SourceKey(row["repo"], path) for path in python_paths]


def scan_sources(
    paths: Sequence[Path], batch_size: int, limit: int | None
) -> tuple[set[SourceKey], Counter]:
    source_keys: set[SourceKey] = set()
    counts: Counter = Counter()
    columns = ["instance_id", "repo", "patch", "problem_statement"]
    for row in iter_raw_rows(paths, columns, batch_size, limit):
        counts["raw"] += 1
        reason, keys = classify_ineligible(row)
        if reason:
            counts[reason] += 1
            continue
        counts["eligible_before_extraction"] += 1
        source_keys.update(keys)
    return source_keys, counts


def prefetch_sources(
    keys: Iterable[SourceKey], cache_dir: Path, workers: int, retries: int
) -> tuple[set[SourceKey], dict[SourceKey, str]]:
    ordered = sorted(set(keys), key=lambda key: (key.repo, key.path))
    failures: dict[SourceKey, str] = {}
    available: set[SourceKey] = set()
    if not ordered:
        return available, failures

    print(f"Prefetching {len(ordered):,} unique source files with {workers} workers...")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(fetch_source, key, cache_dir, retries): key for key in ordered
        }
        for completed, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            try:
                future.result()
                available.add(key)
            except Exception as exc:  # keep the full build moving and report failures
                failures[key] = str(exc)
            if completed % 250 == 0 or completed == len(ordered):
                print(
                    f"  sources {completed:,}/{len(ordered):,}; "
                    f"failed={len(failures):,}",
                    flush=True,
                )
    return available, failures


def write_processed_rows(
    paths: Sequence[Path],
    cache_dir: Path,
    output_path: Path,
    batch_size: int,
    limit: int | None,
) -> tuple[Counter, dict[str, list[str]]]:
    counts: Counter = Counter()
    examples: dict[str, list[str]] = {}
    pending: list[dict] = []
    source_materials: dict[SourceKey, tuple[str, SourceAnalysis]] = {}
    writer = pq.ParquetWriter(output_path, OUTPUT_SCHEMA, compression="zstd")
    columns = ["instance_id", "repo", "patch", "problem_statement"]

    def skip(reason: str, instance_id: str) -> None:
        counts[reason] += 1
        examples.setdefault(reason, [])
        if len(examples[reason]) < 20:
            examples[reason].append(instance_id)

    try:
        for row in iter_raw_rows(paths, columns, batch_size, limit):
            counts["raw"] += 1
            reason, _ = classify_ineligible(row)
            if reason:
                skip(reason, row["instance_id"])
                continue
            try:
                pending.append(process_instance(row, cache_dir, source_materials))
            except SourceFetchError as exc:
                skip("source_fetch_error", row["instance_id"])
                examples.setdefault("source_fetch_error_detail", [])
                if len(examples["source_fetch_error_detail"]) < 20:
                    examples["source_fetch_error_detail"].append(str(exc))
                continue
            except PatchApplyError as exc:
                skip("patch_apply_error", row["instance_id"])
                examples.setdefault("patch_apply_error_detail", [])
                if len(examples["patch_apply_error_detail"]) < 20:
                    examples["patch_apply_error_detail"].append(str(exc))
                continue
            except SourceParseError as exc:
                skip("source_parse_error", row["instance_id"])
                examples.setdefault("source_parse_error_detail", [])
                if len(examples["source_parse_error_detail"]) < 20:
                    examples["source_parse_error_detail"].append(str(exc))
                continue
            except BuildError as exc:
                skip("processing_error", row["instance_id"])
                examples.setdefault("processing_error_detail", [])
                if len(examples["processing_error_detail"]) < 20:
                    examples["processing_error_detail"].append(str(exc))
                continue

            counts["kept"] += 1
            if len(pending) >= batch_size:
                writer.write_table(pa.Table.from_pylist(pending, schema=OUTPUT_SCHEMA))
                pending.clear()
            if counts["raw"] % 2_000 == 0:
                print(
                    f"  rows {counts['raw']:,}; kept={counts['kept']:,}; "
                    f"skipped={counts['raw'] - counts['kept']:,}",
                    flush=True,
                )
        if pending:
            writer.write_table(pa.Table.from_pylist(pending, schema=OUTPUT_SCHEMA))
    finally:
        writer.close()
    return counts, examples


def split_dataset(
    all_path: Path, train_path: Path, validation_path: Path, validation_size: int, seed: int
) -> tuple[int, int]:
    table = pq.read_table(all_path)
    if validation_size < 0 or validation_size >= table.num_rows:
        raise ValueError(
            f"validation-size must be in [0, {max(table.num_rows - 1, 0)}], "
            f"got {validation_size}"
        )
    indices = list(range(table.num_rows))
    random.Random(seed).shuffle(indices)
    shuffled = table.take(pa.array(indices, type=pa.int64()))
    split_at = table.num_rows - validation_size
    train = shuffled.slice(0, split_at)
    validation = shuffled.slice(split_at, validation_size)
    pq.write_table(train, train_path, compression="zstd")
    pq.write_table(validation, validation_path, compression="zstd")
    return train.num_rows, validation.num_rows


def ensure_output(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [
        output_dir / "all.parquet",
        output_dir / "train.parquet",
        output_dir / "validation.parquet",
        output_dir / "cleaning_report.json",
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists ({joined}); pass --overwrite")
    if overwrite:
        for path in existing:
            if path.is_file():
                path.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = parquet_files(args.input)
    ensure_output(args.output, args.overwrite)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {len(paths)} raw parquet shard(s)...")
    source_keys, scan_counts = scan_sources(paths, args.batch_size, args.limit)
    print(
        f"Raw={scan_counts['raw']:,}; "
        f"eligible_before_extraction={scan_counts['eligible_before_extraction']:,}; "
        f"unique_sources={len(source_keys):,}"
    )
    _, fetch_failures = prefetch_sources(
        source_keys, args.cache_dir, args.workers, args.fetch_retries
    )

    staging = args.output / "all.parquet"
    print("Extracting file/module/function targets...")
    counts, examples = write_processed_rows(
        paths, args.cache_dir, staging, args.batch_size, args.limit
    )
    train_rows, validation_rows = split_dataset(
        staging,
        args.output / "train.parquet",
        args.output / "validation.parquet",
        args.validation_size,
        args.seed,
    )

    report = {
        "input": [str(path) for path in paths],
        "output": str(args.output),
        "seed": args.seed,
        "validation_size": args.validation_size,
        "limit": args.limit,
        "counts": dict(sorted(counts.items())),
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "unique_source_files": len(source_keys),
        "source_fetch_failures": len(fetch_failures),
        "source_fetch_failure_examples": [
            {"repo": key.repo, "path": key.path, "error": error}
            for key, error in list(fetch_failures.items())[:20]
        ],
        "skip_examples": examples,
    }
    (args.output / "cleaning_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not args.keep_all:
        staging.unlink()

    print(
        f"Done: train={train_rows:,}, validation={validation_rows:,}, "
        f"total={train_rows + validation_rows:,} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
