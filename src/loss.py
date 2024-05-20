import torch


def mean_log_prob_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Loss function for the transformer model.

    Args:
        logits (torch.Tensor): Logits output by the model.
        labels (torch.Tensor): True labels, in this case the sum of the first two numbers modulo p.

    Returns:
        torch.Tensor: the negative mean
    """
    if len(logits.shape) == 3:
        logits = logits[:, -1]
    logits = logits.to(torch.float64)
    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
    return -correct_log_probs.mean()
