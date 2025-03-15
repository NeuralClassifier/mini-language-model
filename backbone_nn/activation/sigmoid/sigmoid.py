import torch

def sigmoid(x):
    """
    Sigmoid
    
    sigmoid(x) = 1 / (1 + exp(-x))
    """
    return 1 / (1 + torch.exp(-x))