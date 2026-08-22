import json
import os

from src.utils.trajectory_tokens import build_assistant_loss_mask


def validate_loss_mask(messages):
    token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
    _, response_ids, loss_mask = build_assistant_loss_mask(token_messages)
    assert len(response_ids) == len(loss_mask), (
        f"Response ids length {len(response_ids)} != loss mask length {len(loss_mask)}"
    )
    return response_ids, loss_mask


def test_loss_mask_masks_observations():
    messages = [
        {"kind": "TokenEvent", "prompt_token_ids": [1], "response_token_ids": [2]},
        {
            "kind": "TokenEvent",
            "prompt_token_ids": [1, 2, 3],
            "response_token_ids": [4],
        },
    ]
    assert validate_loss_mask(messages) == ([2, 3, 4], [1, 0, 1])


if __name__ == "__main__":
    import argparse

    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--messages_path",
        type=str,
        required=True,
        help="Path to the messages JSON file",
    )
    args = parser.parse_args()

    messages_files = os.listdir(args.messages_path)
    for file_name in messages_files:
        if not file_name.endswith(".json"):
            continue
        print(f"Processing file: {file_name}")
        full_path = os.path.join(args.messages_path, file_name)
        with open(full_path, "r") as f:
            messages = json.load(f)["messages"]

        try:
            response_ids, loss_mask = validate_loss_mask(messages)
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
            with open("debug_response_ids.txt", "w") as output:
                for mask, token_id in zip(loss_mask, response_ids):
                    output.write(
                        f"Token ID: {token_id}, Mask: {mask}, "
                        f"Token: {tokenizer.decode([token_id])}\n"
                    )
        except Exception as e:
            print(f"Error processing {full_path}: {e}")

        # break
