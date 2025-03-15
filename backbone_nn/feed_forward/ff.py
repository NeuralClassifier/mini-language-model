import torch
import torch.nn as nn
from linear.lin import Linear
from softmax.softm import softmax
from activation.relu.relu import relu
import numpy as np


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.linear1 = Linear(embed_dim, hidden_dim)
        self.linear2 = Linear(hidden_dim, embed_dim)
    
    def forward(self, x):
        # FFN: x -> linear1 -> ReLU -> linear2
        return self.linear2(relu(self.linear1(x)))