#!/usr/bin/env python3
"""Split real source samples by rollout-derived task difficulty.

The rollout parquet is used only to compute an instance-level difficulty label.
Its conversations and reward payloads are never copied to the trainable output
partitions. Each partition preserves the source parquet schema exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

DIFFICULTY_ORDER = ("easy", "medium", "hard")
REWARD_COMPONENTS = ("file_reward", "module_reward", "entity_reward")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate repeated rollouts, calibrate for training step, and split "
            "the matching real source samples into easy/medium/hard parquets."
        )
    )
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage-window",
        type=int,
        default=25,
        help="Number of training steps per local difficulty calibration window.",
    )
    parser.add_argument(
        "--easy-perfect-rate",
        type=float,
        default=0.75,
        help="Minimum exact-success rate required for an easy sample.",
    )
    parser.add_argument(
        "--easy-mean-score",
        type=float,
        default=0.85,
        help="Minimum mean normalized reward required for an easy sample.",
    )
    parser.add_argument(
        "--hard-quantile",
        type=float,
        default=0.75,
        help=(
            "Within-stage difficulty quantile used as the hard cutoff; hard "
            "samples must also have zero exact successes."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stable_fraction(value: str) -> float:
    """Return a deterministic [0, 1) tie breaker without global RNG state."""
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def linear_quantile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute a quantile of an empty sequence")
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def normalized_reward(reward_dict: dict[str, Any]) -> float:
    """Normalize the three localization reward components to [0, 1]."""
    if not isinstance(reward_dict, dict):
        raise TypeError("reward_dict must be a mapping")
    values = []
    for field in REWARD_COMPONENTS:
        value = reward_dict.get(field)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"Missing or non-finite reward component: {field}")
        value = float(value)
        if not -1e-9 <= value <= 1.0 + 1e-9:
            raise ValueError(f"Reward component {field} is outside [0, 1]: {value}")
        values.append(min(1.0, max(0.0, value)))

    component_sum = sum(values)
    recorded_sum = reward_dict.get("multilevel_localization_f1_reward")
    if recorded_sum is not None and not math.isclose(
        component_sum, float(recorded_sum), rel_tol=1e-7, abs_tol=1e-7
    ):
        raise ValueError(
            "multilevel_localization_f1_reward does not equal its components"
        )
    return component_sum / len(REWARD_COMPONENTS)


def aggregate_rollouts(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        instance_id = row.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError("Every rollout must have a non-empty instance_id")
        grouped[instance_id].append(row)

    aggregates = []
    for instance_id, instance_rows in grouped.items():
        steps = {int(row["step"]) for row in instance_rows}
        if len(steps) != 1:
            raise ValueError(
                f"{instance_id} appears at multiple training steps: {sorted(steps)}"
            )
        rollout_numbers = [int(row["rollout_number"]) for row in instance_rows]
        if len(rollout_numbers) != len(set(rollout_numbers)):
            raise ValueError(f"{instance_id} has duplicate rollout_number values")

        scores = [normalized_reward(row["reward_dict"]) for row in instance_rows]
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        perfect_count = sum(math.isclose(score, 1.0, abs_tol=1e-9) for score in scores)
        aggregates.append(
            {
                "instance_id": instance_id,
                "step": steps.pop(),
                "rollout_count": len(scores),
                "mean_normalized_reward": mean_score,
                "reward_std": math.sqrt(variance),
                "perfect_count": perfect_count,
                "perfect_rate": perfect_count / len(scores),
                "raw_difficulty": 1.0 - mean_score,
            }
        )
    return aggregates


def label_difficulty(
    aggregates: list[dict[str, Any]],
    *,
    stage_window: int,
    easy_perfect_rate: float,
    easy_mean_score: float,
    hard_quantile: float,
) -> list[dict[str, Any]]:
    if stage_window <= 0:
        raise ValueError("stage_window must be positive")
    for name, value in (
        ("easy_perfect_rate", easy_perfect_rate),
        ("easy_mean_score", easy_mean_score),
        ("hard_quantile", hard_quantile),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if not aggregates:
        return []

    minimum_step = min(row["step"] for row in aggregates)
    by_stage: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregates:
        stage = (row["step"] - minimum_step) // stage_window
        row["stage"] = stage
        row["stage_start_step"] = minimum_step + stage * stage_window
        row["stage_end_step"] = row["stage_start_step"] + stage_window - 1
        by_stage[stage].append(row)

    for stage_rows in by_stage.values():
        cutoff = linear_quantile(
            (row["raw_difficulty"] for row in stage_rows), hard_quantile
        )
        ordered = sorted(
            stage_rows,
            key=lambda row: (
                row["raw_difficulty"],
                stable_fraction(row["instance_id"]),
            ),
        )
        denominator = max(1, len(ordered) - 1)
        for rank, row in enumerate(ordered):
            row["stage_difficulty_percentile"] = rank / denominator
            row["stage_hard_cutoff"] = cutoff
            if (
                row["perfect_rate"] >= easy_perfect_rate
                and row["mean_normalized_reward"] >= easy_mean_score
            ):
                row["difficulty"] = "easy"
            elif row["perfect_count"] == 0 and row["raw_difficulty"] >= cutoff:
                row["difficulty"] = "hard"
            else:
                row["difficulty"] = "medium"

    return aggregates


def prepare_output_directory(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    managed_names = {
        *(f"{name}.parquet" for name in DIFFICULTY_ORDER),
        "unmatched.parquet",
        "difficulty_index.parquet",
        "report.json",
    }
    existing = [path / name for name in managed_names if (path / name).exists()]
    if existing and not overwrite:
        names = ", ".join(sorted(item.name for item in existing))
        raise FileExistsError(
            f"Output files already exist ({names}); pass --overwrite to replace them"
        )


def split_source_table(
    source: pa.Table, labels: dict[str, str]
) -> dict[str, pa.Table]:
    if "instance_id" not in source.column_names:
        raise ValueError("Source parquet must contain an instance_id column")
    instance_ids = source["instance_id"].to_pylist()
    partitions: dict[str, pa.Table] = {}
    for difficulty in DIFFICULTY_ORDER:
        mask = pa.array([labels.get(value) == difficulty for value in instance_ids])
        partitions[difficulty] = source.filter(mask)
    unmatched_mask = pa.array([value not in labels for value in instance_ids])
    partitions["unmatched"] = source.filter(unmatched_mask)
    return partitions


def index_table(rows: list[dict[str, Any]]) -> pa.Table:
    ordered = sorted(rows, key=lambda row: row["instance_id"])
    fields = (
        "instance_id",
        "difficulty",
        "step",
        "stage",
        "stage_start_step",
        "stage_end_step",
        "rollout_count",
        "mean_normalized_reward",
        "reward_std",
        "perfect_count",
        "perfect_rate",
        "raw_difficulty",
        "stage_difficulty_percentile",
        "stage_hard_cutoff",
    )
    return pa.Table.from_pylist(
        [{field: row[field] for field in fields} for row in ordered]
    )


def main() -> None:
    args = parse_args()
    prepare_output_directory(args.output, args.overwrite)

    rollout_columns = ["instance_id", "reward_dict", "step", "rollout_number"]
    rollout_table = pq.read_table(args.rollouts, columns=rollout_columns)
    aggregates = aggregate_rollouts(rollout_table.to_pylist())
    aggregates = label_difficulty(
        aggregates,
        stage_window=args.stage_window,
        easy_perfect_rate=args.easy_perfect_rate,
        easy_mean_score=args.easy_mean_score,
        hard_quantile=args.hard_quantile,
    )

    source = pq.read_table(args.source)
    source_ids = set(source["instance_id"].to_pylist())
    matched = [row for row in aggregates if row["instance_id"] in source_ids]
    labels = {row["instance_id"]: row["difficulty"] for row in matched}
    partitions = split_source_table(source, labels)

    for name, table in partitions.items():
        pq.write_table(table, args.output / f"{name}.parquet", compression="zstd")
    pq.write_table(
        index_table(matched),
        args.output / "difficulty_index.parquet",
        compression="zstd",
    )

    difficulty_counts = {
        name: partitions[name].num_rows for name in DIFFICULTY_ORDER
    }
    report = {
        "rollouts_path": str(args.rollouts),
        "source_path": str(args.source),
        "output_path": str(args.output),
        "rollout_rows": rollout_table.num_rows,
        "rollout_instances": len(aggregates),
        "source_rows": source.num_rows,
        "matched_source_rows": len(matched),
        "rollout_instances_absent_from_source": len(aggregates) - len(matched),
        "unmatched_source_rows": partitions["unmatched"].num_rows,
        "difficulty_counts": difficulty_counts,
        "stage_window": args.stage_window,
        "easy_perfect_rate": args.easy_perfect_rate,
        "easy_mean_score": args.easy_mean_score,
        "hard_quantile": args.hard_quantile,
        "reward_normalization": "mean(file_reward,module_reward,entity_reward)",
        "train_partitions_preserve_source_schema": all(
            partitions[name].schema == source.schema for name in partitions
        ),
        "rollout_messages_copied_to_train_partitions": False,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
