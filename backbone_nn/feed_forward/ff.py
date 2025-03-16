import torch
import torch.nn as nn

import sys
import os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

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