import math

import pyarrow as pa

from scripts.split_rollout_difficulty import (
    aggregate_rollouts,
    label_difficulty,
    normalized_reward,
    split_source_table,
)


def _rollout(instance_id, step, rollout_number, score):
    return {
        "instance_id": instance_id,
        "step": step,
        "rollout_number": rollout_number,
        "reward_dict": {
            "multilevel_localization_f1_reward": score * 3,
            "file_reward": score,
            "module_reward": score,
            "entity_reward": score,
        },
    }


def test_normalized_reward_uses_only_localization_components():
    reward = {
        "multilevel_localization_f1_reward": 1.5,
        "file_reward": 1.0,
        "module_reward": 0.5,
        "entity_reward": 0.0,
        "multiturn_reward": 100.0,
    }

    assert normalized_reward(reward) == 0.5


def test_aggregate_rollouts_computes_instance_level_statistics():
    rows = [
        _rollout("task", 9, 0, 1.0),
        _rollout("task", 9, 1, 0.5),
        _rollout("task", 9, 2, 0.0),
        _rollout("task", 9, 3, 1.0),
    ]

    result = aggregate_rollouts(rows)[0]

    assert result["rollout_count"] == 4
    assert result["perfect_count"] == 2
    assert result["perfect_rate"] == 0.5
    assert result["mean_normalized_reward"] == 0.625
    assert math.isclose(result["reward_std"], math.sqrt(0.171875))


def test_labels_easy_hard_and_boundary_medium_within_stage():
    aggregates = []
    for instance_id, mean, perfect_count in [
        ("easy", 0.95, 4),
        ("boundary", 0.70, 2),
        ("partial", 0.45, 0),
        ("hard", 0.10, 0),
    ]:
        aggregates.append(
            {
                "instance_id": instance_id,
                "step": 10,
                "rollout_count": 4,
                "mean_normalized_reward": mean,
                "reward_std": 0.0,
                "perfect_count": perfect_count,
                "perfect_rate": perfect_count / 4,
                "raw_difficulty": 1.0 - mean,
            }
        )

    result = label_difficulty(
        aggregates,
        stage_window=25,
        easy_perfect_rate=0.75,
        easy_mean_score=0.85,
        hard_quantile=0.75,
    )
    labels = {row["instance_id"]: row["difficulty"] for row in result}

    assert labels == {
        "easy": "easy",
        "boundary": "medium",
        "partial": "medium",
        "hard": "hard",
    }


def test_source_partitions_keep_schema_and_exclude_rollout_payloads():
    source = pa.Table.from_pylist(
        [
            {"instance_id": "a", "prompt": "A"},
            {"instance_id": "b", "prompt": "B"},
            {"instance_id": "c", "prompt": "C"},
            {"instance_id": "d", "prompt": "D"},
        ]
    )

    partitions = split_source_table(
        source, {"a": "easy", "b": "medium", "c": "hard"}
    )

    assert partitions["easy"]["instance_id"].to_pylist() == ["a"]
    assert partitions["medium"]["instance_id"].to_pylist() == ["b"]
    assert partitions["hard"]["instance_id"].to_pylist() == ["c"]
    assert partitions["unmatched"]["instance_id"].to_pylist() == ["d"]
    assert all(table.schema == source.schema for table in partitions.values())
    assert all(
        "chat_messages" not in table.column_names for table in partitions.values()
    )
