"""Small sampling utilities used during generation."""

import torch


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("inf")):
    """Apply top-k and/or nucleus filtering to a batch of logits.

    Args:
        logits: Tensor of token logits with vocabulary dimension last.
        top_k: Keep only the top-k logits if greater than zero.
        top_p: Keep the smallest prefix of tokens whose cumulative probability is
            at most `top_p`.
        filter_value: Value assigned to filtered logits.

    Returns:
        A logits tensor with the filtered entries replaced by `filter_value`.
    """
    top_k = min(top_k, logits.size(-1))

    if top_k > 0:
        threshold = torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(logits < threshold, filter_value)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # Remove the tail of the distribution once the cumulative probability budget is exceeded.
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits
