import torch

def softplus(x):
    """
    Softplus
    
    softplus(x) = log(1 + exp(x))
    """
    return torch.log1p(torch.exp(x))