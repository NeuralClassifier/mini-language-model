import torch

def elu(x, alpha=1.0):
    """
    Exponential Linear Unit
    
    For x > 0, ELU(x) = x, and for x <= 0, ELU(x) = alpha * (exp(x) - 1).
    """
    return torch.where(x > 0, x, alpha * (torch.exp(x) - 1))
