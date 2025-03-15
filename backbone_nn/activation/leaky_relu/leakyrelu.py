import torch

def custom_leaky_relu(x, negative_slope=0.01):
    """
    Leaky ReLU
    
    Retunrs x for positive values and negative_slope x x for negative values
    """
    return torch.where(x > 0, x, negative_slope * x)