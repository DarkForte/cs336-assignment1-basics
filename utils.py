import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Computes the softmax of the input tensor along the specified dimension.
    """
    max_x = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_x)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x

def cross_entropy_loss(logits, labels):
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    logits = logits - max_logits
    labels.unsqueeze_(-1)  # Ensure labels are of shape (batch, seq, 1)
    # Gather the logits corresponding to the labels
    correct_logits = logits.gather(dim=-1, index=labels.long()).squeeze(-1)  # Shape: (batch, seq)
    log_probs = torch.log(torch.exp(logits).sum(dim=-1))
    return (-correct_logits + log_probs).mean()


def gradient_clipping(params, max_norm, eps=1e-6):
    for p in params:
        if p.grad is None:
            continue
        l2_norm = p.grad.data.norm(2)
        print("l2_norm: ", l2_norm)
        if l2_norm > max_norm:
            scale = max_norm / (l2_norm + eps)
            p.grad.data *= scale