import torch

def softmax(scores, dim=-1):
    # Subtract the maximum value for numerical stability
    max_scores, _ = torch.max(scores, dim=dim, keepdim=True)
    exp_scores = torch.exp(scores - max_scores)
    sum_exp_scores = torch.sum(exp_scores, dim=dim, keepdim=True)
    return exp_scores / sum_exp_scores