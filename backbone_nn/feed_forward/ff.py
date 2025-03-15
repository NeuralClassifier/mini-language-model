import torch
import torch.nn as nn
from vocab_mapping.vocab_mapping import vocabulary_mapping
from backbone_nn.linear.lin import Linear
from backbone_nn.softmax.softm import softmax
import numpy as np


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
    
    def forward(self, x):
        # FFN: x -> linear1 -> ReLU -> linear2
        return self.linear2(F.relu(self.linear1(x)))