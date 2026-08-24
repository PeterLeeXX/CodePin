from collections import Counter

import pyarrow as pa

from scripts.select_sft_candidates import (
    attach_metadata,
    boundary_score,
    mutation_type,
    select_candidates,
    selection_quotas,
    source_table,
)


def _source(index):
    return {
        "instance_id": f"repo.commit.mutation__{index}",
        "repo": "repo" if index < 16 else "rare-repo",
        "prompt": [{"role": "user", "content": f"question {index}"}],
        "target": [{"file": f"file_{index}.py"}],
    }


def _difficulty(index):
    mean = (index % 8 + 1) / 9
    perfect_count = index % 4
    return {
        "instance_id": _source(index)["instance_id"],
        "difficulty": "medium" if index < 12 else "hard",
        "step": index + 1,
        "stage": 0,
        "rollout_count": 4,
        "mean_normalized_reward": mean,
        "reward_std": 0.2,
        "perfect_count": perfect_count,
        "perfect_rate": perfect_count / 4,
        "raw_difficulty": 1 - mean,
        "stage_difficulty_percentile": index / 20,
    }


def test_selection_quotas_are_exact():
    assert selection_quotas(6000, 0.60, 0.25) == {
        "representative": 3600,
        "boundary": 1500,
        "long_tail": 900,
    }


def test_mutation_type_parses_from_right_across_repo_formats():
    assert (
        mutation_type(
            "pdfminer__pdfminer.six.1a8bd2f7.func_basic__ze2uf0zg",
            "swesmith/pdfminer__pdfminer.six.1a8bd2f7",
        )
        == "func_basic"
    )
    assert (
        mutation_type(
            "Project-MONAI__MONAI.a09c1f08.pr_3560",
            "swesmith/Project-MONAI__MONAI.a09c1f08",
        )
        == "pr"
    )


def test_boundary_score_prefers_uncertain_over_saturated_sample():
    uncertain = {
        "mean_normalized_reward": 0.5,
        "perfect_rate": 0.5,
        "reward_std": 0.3,
    }
    saturated = {
        "mean_normalized_reward": 1.0,
        "perfect_rate": 1.0,
        "reward_std": 0.0,
    }

    assert boundary_score(uncertain) > boundary_score(saturated)


def test_selection_is_disjoint_deterministic_and_preserves_schema():
    sources = [_source(index) for index in range(20)]
    difficulties = [_difficulty(index) for index in range(20)]
    candidates, dropped = attach_metadata(sources, difficulties, seed=42)
    quotas = {"representative": 6, "boundary": 3, "long_tail": 1}

    first = select_candidates(candidates, quotas, seed=42)
    second_candidates, _ = attach_metadata(sources, difficulties, seed=42)
    second = select_candidates(second_candidates, quotas, seed=42)

    assert dropped == 0
    assert [row["instance_id"] for row in first] == [
        row["instance_id"] for row in second
    ]
    assert len({row["instance_id"] for row in first}) == 10
    assert Counter(row["selection_bucket"] for row in first) == quotas

    schema = pa.Table.from_pylist(sources).schema
    table = source_table(first, schema)
    assert table.schema == schema
    assert "difficulty" not in table.column_names
    assert "reward_std" not in table.column_names


def test_prompt_target_dedup_keeps_one_copy():
    sources = [_source(0), {**_source(0), "instance_id": "repo.commit.mutation__copy"}]
    difficulties = [
        _difficulty(0),
        {**_difficulty(0), "instance_id": "repo.commit.mutation__copy"},
    ]

    candidates, dropped = attach_metadata(sources, difficulties, seed=42)

    assert len(candidates) == 1
    assert dropped == 1
