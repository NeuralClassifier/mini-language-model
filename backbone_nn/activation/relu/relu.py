import torch

def relu(x):
    """
    Rectified Linear Unit
    
    relu(x) = max(0, x)
    """

    zeros = torch.zeros_like(x)
    # if x > 0, return x otherwise, retun 0
    return torch.where(x > 0, x, zeros)