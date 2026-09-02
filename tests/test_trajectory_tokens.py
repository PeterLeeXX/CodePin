import pytest

from src.utils.trajectory_tokens import build_assistant_loss_mask


def test_build_assistant_loss_mask_masks_tool_observations():
    events = [
        {"prompt_token_ids": [1, 2], "response_token_ids": [3, 4]},
        {"prompt_token_ids": [1, 2, 3, 4, 5, 6], "response_token_ids": [7]},
    ]

    prompt, response, mask = build_assistant_loss_mask(events)

    assert prompt == [1, 2]
    assert response == [3, 4, 5, 6, 7]
    assert mask == [1, 1, 0, 0, 1]


def test_build_assistant_loss_mask_rejects_broken_prefix():
    events = [
        {"prompt_token_ids": [1], "response_token_ids": [2]},
        {"prompt_token_ids": [9, 2], "response_token_ids": [3]},
    ]

    with pytest.raises(ValueError, match="does not extend"):
        build_assistant_loss_mask(events)


def test_build_assistant_loss_mask_rejects_reordered_or_empty_turns():
    events = [
        {"prompt_token_ids": [1], "response_token_ids": [2]},
        {"prompt_token_ids": [1], "response_token_ids": [2]},
    ]
    with pytest.raises(ValueError, match="out of order"):
        build_assistant_loss_mask(events)
    with pytest.raises(ValueError, match="empty"):
        build_assistant_loss_mask([{"prompt_token_ids": [1], "response_token_ids": []}])
