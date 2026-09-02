"""Run real model trajectories through cleaning, scoring, export and SkyRL tokenization.

This script never starts training. It fails if the real pipeline produces no
usable SFT examples, if SkyRL cannot read an export, or if optional judging fails.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from src.data_pipeline import (
    clean_tasks,
    export_data,
    generate_trajectories,
    load_rows,
)
from src.evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--judge", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tasks, cleaning = clean_tasks(load_rows(args.tasks)[: args.limit])
    (args.output / "cleaning.json").write_text(
        json.dumps(cleaning, indent=2), encoding="utf-8"
    )
    if not tasks:
        raise RuntimeError("No valid input tasks")
    (args.output / "tasks.jsonl").write_text(
        "".join(json.dumps(t) + "\n" for t in tasks), encoding="utf-8"
    )
    trajectories = generate_trajectories(
        tasks,
        args.output / "trajectories",
        model="openai/codepin",
        base_url=args.base_url,
        concurrency=2,
        max_turns=8,
    )
    exported = export_data(
        tasks,
        trajectories,
        args.output / "export",
        validation_fraction=0,
        min_quality=0.5,
    )
    report = evaluate(
        tasks,
        trajectories,
        judge={"base_url": args.base_url, "model": "codepin"} if args.judge else None,
    )
    report["cleaning"] = cleaning
    report["export"] = exported
    (args.output / "evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if not exported["sft_kept"]:
        raise RuntimeError(
            "No real trajectories passed SFT quality filtering; see evaluation.json"
        )

    from skyrl.train.config.sft_config import TrainOnWhat
    from skyrl.train.dataset.dataset import PromptDataset
    from skyrl.train.sft_trainer import collate_sft_batch, tokenize_chat_example

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = pq.read_table(args.output / "export/sft/train.parquet").to_pylist()
    tokenized = [
        tokenize_chat_example(
            row,
            tokenizer,
            max_length=16384,
            train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        )
        for row in rows
    ]
    if any(row is None for row in tokenized):
        raise RuntimeError("SkyRL rejected an exported SFT example")
    for row in tokenized:
        assert len(row["input_ids"]) == len(row["attention_mask"])
        assert 0 < sum(row["loss_mask"]) <= row["num_actions"]
        assert 0 in row["loss_mask"], "tool observations must be masked"
    batch = collate_sft_batch(tokenized, tokenizer)
    rl_dataset = PromptDataset(
        str(args.output / "export/rl/train.parquet"),
        tokenizer,
        max_prompt_length=16384,
        num_workers=1,
    )
    assert len(rl_dataset) == len(tasks)
    assert rl_dataset[0][2]["file_changes"]
    # Exercise the actual SkyRL collation object without creating a trainer.
    summary = {
        "sft_rows": len(rows),
        "batch_type": type(batch).__name__,
        "supervised_tokens": [sum(row["loss_mask"]) for row in tokenized],
        "rl_rows": len(rl_dataset),
    }
    (args.output / "skyrl_readback.json").write_text(json.dumps(summary, indent=2))

    import ray
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        RemoteInferenceClient,
    )
    from skyrl.train.generators.base import BatchMetadata, TrajectoryID

    from src.generator.code_search_generator import CodeSearchGenerator
    from src.train import CodeSearchGeneratorConfig

    endpoint = args.base_url.removesuffix("/v1")
    client = RemoteInferenceClient(endpoint, [endpoint], 1, model_name="codepin")
    cfg = CodeSearchGeneratorConfig(
        max_turns=8,
        max_train_length=16384,
        traj_dir=str(args.output / "skyrl_trajectories"),
    )
    cfg.sampling_params.max_generate_length = 2048
    cfg.sampling_params.temperature = 0.0
    accepted = next(t for t in tasks if t["instance_id"] == rows[0]["instance_id"])
    batch_input = {
        "prompts": [accepted["prompt"]],
        "env_extras": [accepted],
        "trajectory_ids": [TrajectoryID(accepted["instance_id"], 0)],
        "batch_metadata": BatchMetadata(0, "eval"),
    }
    ray.init(num_cpus=2, include_dashboard=False, object_store_memory=256 * 1024**2)
    try:
        generator = CodeSearchGenerator(cfg, client, tokenizer, "codepin")
        generated = asyncio.run(generator.generate(batch_input))
        assert generated["stop_reasons"] == ["stop"], generated["stop_reasons"]
        assert sum(generated["loss_masks"][0]) > 0
        assert 0 in generated["loss_masks"][0]
        cfg.max_turns = 1
        cfg.sampling_params.max_generate_length = 8
        batch_input["batch_metadata"] = BatchMetadata(1, "eval")
        invalid = asyncio.run(generator.generate(batch_input))
        assert invalid["stop_reasons"][0] != "stop"
        assert invalid["rewards"] == [0.0]
        assert not any(invalid["loss_masks"][0])
        (args.output / "skyrl_rollout.json").write_text(
            json.dumps(
                {
                    "valid_stop": generated["stop_reasons"],
                    "valid_reward": generated["rewards"],
                    "valid_loss_tokens": sum(generated["loss_masks"][0]),
                    "invalid_stop": invalid["stop_reasons"],
                    "invalid_reward": invalid["rewards"],
                    "invalid_loss_tokens": sum(invalid["loss_masks"][0]),
                },
                indent=2,
            )
        )
    finally:
        ray.shutdown()
    if args.judge and report["judge"]["failed"]:
        raise RuntimeError("LLM judge failed; see evaluation.json")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
