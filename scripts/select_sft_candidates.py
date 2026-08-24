#!/usr/bin/env python3
"""Select a first-round teacher-generation pool for text SFT.

The selector consumes real source samples plus the rollout-derived difficulty
index. It emits source-schema-only parquet files; rollout conversations and
reward payloads remain outside the files intended for teacher generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SELECTION_BUCKETS = ("representative", "boundary", "long_tail")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select disjoint representative, boundary, and long-tail source "
            "samples for first-round teacher data generation."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--difficulty-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=6000)
    parser.add_argument("--representative-ratio", type=float, default=0.60)
    parser.add_argument("--boundary-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stable_key(value: str, seed: int, namespace: str) -> int:
    payload = f"{namespace}:{seed}:{value}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def mutation_type(instance_id: str, repo: str) -> str:
    del repo  # The source repo field includes a swesmith/ prefix and commit.
    if "." not in instance_id:
        return "unknown"
    final_component = instance_id.rsplit(".", 1)[-1]
    if final_component.startswith("pr_"):
        return "pr"
    return final_component.rsplit("__", 1)[0] or "unknown"


def task_fingerprint(row: dict[str, Any]) -> str:
    payload = {"prompt": row.get("prompt"), "target": row.get("target")}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def selection_quotas(
    size: int, representative_ratio: float, boundary_ratio: float
) -> dict[str, int]:
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0.0 <= representative_ratio <= 1.0:
        raise ValueError("representative_ratio must be in [0, 1]")
    if not 0.0 <= boundary_ratio <= 1.0:
        raise ValueError("boundary_ratio must be in [0, 1]")
    if representative_ratio + boundary_ratio > 1.0:
        raise ValueError("representative_ratio + boundary_ratio must not exceed 1")

    representative = round(size * representative_ratio)
    boundary = round(size * boundary_ratio)
    long_tail = size - representative - boundary
    return {
        "representative": representative,
        "boundary": boundary,
        "long_tail": long_tail,
    }


def boundary_score(row: dict[str, Any]) -> float:
    """Prefer learnable, unstable samples near the policy decision frontier."""
    mean_score = float(row["mean_normalized_reward"])
    perfect_rate = float(row["perfect_rate"])
    reward_std = float(row["reward_std"])
    mean_uncertainty = 4.0 * mean_score * (1.0 - mean_score)
    exact_uncertainty = 4.0 * perfect_rate * (1.0 - perfect_rate)
    rollout_instability = min(1.0, 2.0 * reward_std)
    return 0.40 * exact_uncertainty + 0.40 * mean_uncertainty + 0.20 * rollout_instability


def select_boundary(
    rows: list[dict[str, Any]], size: int, seed: int
) -> list[dict[str, Any]]:
    if size > len(rows):
        raise ValueError("Boundary quota exceeds the available candidate pool")
    priority = {"medium": 0, "hard": 1, "easy": 2}
    ordered = sorted(
        rows,
        key=lambda row: (
            priority.get(row["difficulty"], 3),
            -boundary_score(row),
            stable_key(row["instance_id"], seed, "boundary"),
        ),
    )
    return ordered[:size]


def select_long_tail(
    rows: list[dict[str, Any]], size: int, seed: int
) -> list[dict[str, Any]]:
    """Round-robin rare repo × mutation × difficulty strata."""
    if size > len(rows):
        raise ValueError("Long-tail quota exceeds the available candidate pool")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["repo"], row["mutation_type"], row["difficulty"])
        groups[key].append(row)
    for key, group in groups.items():
        group.sort(
            key=lambda row: stable_key(
                row["instance_id"], seed, f"long-tail:{key}"
            )
        )

    repo_frequency = Counter(row["repo"] for row in rows)
    mutation_frequency = Counter(row["mutation_type"] for row in rows)
    keys = sorted(
        groups,
        key=lambda key: (
            len(groups[key]),
            repo_frequency[key[0]],
            mutation_frequency[key[1]],
            stable_key("|".join(key), seed, "long-tail-group"),
        ),
    )

    selected: list[dict[str, Any]] = []
    depth = 0
    while len(selected) < size:
        made_progress = False
        for key in keys:
            group = groups[key]
            if depth < len(group):
                selected.append(group[depth])
                made_progress = True
                if len(selected) == size:
                    break
        if not made_progress:
            raise RuntimeError("Could not fill the long-tail quota")
        depth += 1
    return selected


def proportional_quotas(
    group_sizes: dict[tuple[str, ...], int], size: int, seed: int
) -> dict[tuple[str, ...], int]:
    total = sum(group_sizes.values())
    if size > total:
        raise ValueError("Requested sample is larger than the grouped population")
    exact = {key: size * count / total for key, count in group_sizes.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remaining = size - sum(quotas.values())
    order = sorted(
        group_sizes,
        key=lambda key: (
            -(exact[key] - quotas[key]),
            stable_key("|".join(key), seed, "quota"),
        ),
    )
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def select_representative(
    rows: list[dict[str, Any]], size: int, seed: int
) -> list[dict[str, Any]]:
    """Sample proportionally within difficulty × repository strata."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["difficulty"], row["repo"])].append(row)
    quotas = proportional_quotas(
        {key: len(group) for key, group in groups.items()}, size, seed
    )
    selected = []
    for key, group in groups.items():
        ordered = sorted(
            group,
            key=lambda row: stable_key(
                row["instance_id"], seed, f"representative:{key}"
            ),
        )
        selected.extend(ordered[: quotas[key]])
    if len(selected) != size:
        raise RuntimeError("Representative sampler did not fill its exact quota")
    return selected


def attach_metadata(
    source_rows: Iterable[dict[str, Any]],
    difficulty_rows: Iterable[dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
    difficulty_by_id = {row["instance_id"]: row for row in difficulty_rows}
    candidates = []
    for source_row in source_rows:
        instance_id = source_row["instance_id"]
        difficulty = difficulty_by_id.get(instance_id)
        if difficulty is None:
            continue
        candidate = {
            **difficulty,
            "repo": source_row["repo"],
            "mutation_type": mutation_type(instance_id, source_row["repo"]),
            "task_fingerprint": task_fingerprint(source_row),
            "source_row": source_row,
        }
        candidates.append(candidate)

    # Deduplicate semantically identical prompt/target pairs conservatively.
    candidates.sort(
        key=lambda row: stable_key(row["instance_id"], seed, "deduplicate")
    )
    deduplicated = {}
    for row in candidates:
        deduplicated.setdefault(row["task_fingerprint"], row)
    dropped = len(candidates) - len(deduplicated)
    return list(deduplicated.values()), dropped


def select_candidates(
    candidates: list[dict[str, Any]], quotas: dict[str, int], seed: int
) -> list[dict[str, Any]]:
    if sum(quotas.values()) > len(candidates):
        raise ValueError(
            f"Requested {sum(quotas.values())} samples from only {len(candidates)} candidates"
        )

    boundary = select_boundary(candidates, quotas["boundary"], seed)
    used = {row["instance_id"] for row in boundary}
    remaining = [row for row in candidates if row["instance_id"] not in used]

    long_tail = select_long_tail(remaining, quotas["long_tail"], seed)
    used.update(row["instance_id"] for row in long_tail)
    remaining = [row for row in remaining if row["instance_id"] not in used]

    representative = select_representative(
        remaining, quotas["representative"], seed
    )

    selected = []
    for bucket, rows in (
        ("boundary", boundary),
        ("long_tail", long_tail),
        ("representative", representative),
    ):
        for row in rows:
            row["selection_bucket"] = bucket
            row["boundary_score"] = boundary_score(row)
            selected.append(row)
    return selected


def source_table(rows: Iterable[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    return pa.Table.from_pylist([row["source_row"] for row in rows], schema=schema)


def selection_index(rows: Iterable[dict[str, Any]]) -> pa.Table:
    fields = (
        "instance_id",
        "repo",
        "mutation_type",
        "difficulty",
        "selection_bucket",
        "boundary_score",
        "step",
        "stage",
        "rollout_count",
        "mean_normalized_reward",
        "reward_std",
        "perfect_count",
        "perfect_rate",
        "raw_difficulty",
        "stage_difficulty_percentile",
        "task_fingerprint",
    )
    ordered = sorted(rows, key=lambda row: row["instance_id"])
    return pa.Table.from_pylist(
        [{field: row[field] for field in fields} for row in ordered]
    )


def prepare_output(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    managed = {
        "teacher_generation.parquet",
        "selection_index.parquet",
        "report.json",
        "README.md",
        *(f"{bucket}.parquet" for bucket in SELECTION_BUCKETS),
    }
    existing = sorted(name for name in managed if (path / name).exists())
    if existing and not overwrite:
        raise FileExistsError(
            f"Output files already exist ({', '.join(existing)}); pass --overwrite"
        )


def count_nested(rows: Iterable[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def main() -> None:
    args = parse_args()
    quotas = selection_quotas(
        args.size, args.representative_ratio, args.boundary_ratio
    )
    prepare_output(args.output, args.overwrite)

    source = pq.read_table(args.source)
    difficulty = pq.read_table(args.difficulty_index)
    candidates, duplicates_dropped = attach_metadata(
        source.to_pylist(), difficulty.to_pylist(), args.seed
    )
    selected = select_candidates(candidates, quotas, args.seed)
    selected.sort(
        key=lambda row: stable_key(row["instance_id"], args.seed, "final-order")
    )

    selected_table = source_table(selected, source.schema)
    pq.write_table(
        selected_table,
        args.output / "teacher_generation.parquet",
        compression="zstd",
    )
    for bucket in SELECTION_BUCKETS:
        bucket_rows = [row for row in selected if row["selection_bucket"] == bucket]
        pq.write_table(
            source_table(bucket_rows, source.schema),
            args.output / f"{bucket}.parquet",
            compression="zstd",
        )
    pq.write_table(
        selection_index(selected),
        args.output / "selection_index.parquet",
        compression="zstd",
    )

    report = {
        "source_path": str(args.source),
        "difficulty_index_path": str(args.difficulty_index),
        "output_path": str(args.output),
        "seed": args.seed,
        "source_rows": source.num_rows,
        "difficulty_labeled_rows": difficulty.num_rows,
        "eligible_after_prompt_target_dedup": len(candidates),
        "prompt_target_duplicates_dropped": duplicates_dropped,
        "selected_rows": len(selected),
        "selection_bucket_counts": count_nested(selected, "selection_bucket"),
        "difficulty_counts": count_nested(selected, "difficulty"),
        "repository_coverage": len({row["repo"] for row in selected}),
        "mutation_type_coverage": len({row["mutation_type"] for row in selected}),
        "trainable_outputs_preserve_source_schema": selected_table.schema == source.schema,
        "rollout_messages_or_rewards_copied_to_trainable_outputs": False,
        "note": "These are prompts for teacher generation, not completed SFT conversations.",
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "README.md").write_text(
        "# First-round text SFT teacher-generation pool\n\n"
        f"Selected {len(selected):,} real source prompts with a deterministic "
        f"{quotas['representative']:,}/{quotas['boundary']:,}/"
        f"{quotas['long_tail']:,} representative/boundary/long-tail mix.\n\n"
        "`teacher_generation.parquet` and the three bucket parquets preserve the "
        "source schema. They contain no teacher answer yet and must not be used "
        "as completed SFT conversations. `selection_index.parquet` is audit-only.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
