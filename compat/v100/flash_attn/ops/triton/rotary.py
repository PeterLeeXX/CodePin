"""Import-only fallback for vLLM's optional FlashAttention rotary hook."""


def apply_rotary(x, cos, sin, interleaved=False):
    # V100 uses vLLM's CUDA/native rotary path; this hook is not executed there.
    return x
