from src.rewards import reward

@reward("multiturn_reward")
def multiturn_reward(
    messages,
    maximal_turns=5,
    minimal_turns=1,
    **kwargs
    ) -> float:
    token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
    num_turns = len(token_messages)
    if (num_turns >= minimal_turns) and (num_turns <= maximal_turns):
        return 1.0
    return 0.0