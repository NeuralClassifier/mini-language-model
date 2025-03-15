import torch

def tanh(x):
    """
    Hyperbolic tangent
    
    tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    e_pos = torch.exp(x)
    e_neg = torch.exp(-x)
    return (e_pos - e_neg) / (e_pos + e_neg)