"""Padding helpers used by SkyRL when FlashAttention is disabled."""

import torch


def unpad_input(hidden_states, attention_mask):
    batch_size, seqlen = attention_mask.shape
    indices = torch.nonzero(attention_mask.reshape(-1), as_tuple=False).flatten()
    hidden_states = hidden_states.reshape(batch_size * seqlen, *hidden_states.shape[2:])[indices]
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=attention_mask.device)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
    return hidden_states, indices, cu_seqlens, int(seqlens.max().item()), None


def pad_input(hidden_states, indices, batch, seqlen):
    output = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
    output.index_copy_(0, indices, hidden_states)
    return output.reshape(batch, seqlen, *hidden_states.shape[1:])
