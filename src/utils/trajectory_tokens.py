"""Token-level helpers shared by CodePin SFT and RL pipelines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def build_assistant_loss_mask(
    token_messages: Sequence[dict[str, Any]],
) -> tuple[list[int], list[int], list[int]]:
    """Merge a multi-turn trajectory and supervise assistant tokens only.

    OpenHands records the exact prompt and response token IDs for every model
    turn.  Each later prompt must extend the previous prompt+response.  Using
    that prefix property avoids architecture-specific assumptions about role,
    thinking, or tool-call tokens and therefore works for Qwen3 and Qwen3.5.

    Returns ``(initial_prompt_ids, merged_response_ids, loss_mask)``.
    If the prefix property is violated, ``ValueError`` is raised so callers can
    conservatively discard the trajectory instead of training on observations.
    """

    if not token_messages:
        raise ValueError("at least one TokenEvent is required")

    initial_prompt = list(token_messages[0]["prompt_token_ids"])
    final_prompt = list(token_messages[-1]["prompt_token_ids"])
    final_response = list(token_messages[-1]["response_token_ids"])
    merged = final_prompt + final_response

    if merged[: len(initial_prompt)] != initial_prompt:
        raise ValueError("the final prompt does not extend the initial prompt")

    response = merged[len(initial_prompt) :]
    loss_mask = [0] * len(response)

    for turn_index, message in enumerate(token_messages):
        prompt_ids = list(message["prompt_token_ids"])
        response_ids = list(message["response_token_ids"])

        if merged[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                f"TokenEvent {turn_index} prompt is not a prefix of the final trajectory"
            )

        response_start = len(prompt_ids)
        response_end = response_start + len(response_ids)
        if merged[response_start:response_end] != response_ids:
            raise ValueError(
                f"TokenEvent {turn_index} response is not preserved in the final trajectory"
            )

        relative_start = response_start - len(initial_prompt)
        relative_end = response_end - len(initial_prompt)
        if relative_start < 0 or relative_end > len(response):
            raise ValueError(
                f"TokenEvent {turn_index} lies outside the merged response"
            )
        loss_mask[relative_start:relative_end] = [1] * len(response_ids)

    return initial_prompt, response, loss_mask
