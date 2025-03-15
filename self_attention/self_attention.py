import torch
import torch.nn as nn
import sys
import os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

from vocab_mapping.vocab_mapping import vocabulary_mapping
from backbone_nn.linear.lin import Linear
from backbone_nn.softmax.softm import softmax
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        # We use embed_dim for queries, keys, and values.
        self.W_q = Linear(embed_dim, embed_dim, bias=False)
        self.W_k = Linear(embed_dim, embed_dim, bias=False)
        self.W_v = Linear(embed_dim, embed_dim, bias=False)
    
    def forward(self, x):
        # x shape: [seq_length, embed_dim]
        Q = self.W_q(x)  # [seq_length, embed_dim]
        K = self.W_k(x)  # [seq_length, embed_dim]
        V = self.W_v(x)  # [seq_length, embed_dim]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(0, 1))  # [seq_length, seq_length]
        scores = scores / np.sqrt(self.embed_dim)      # scale by sqrt(d_k)
        
        # Apply softmax to get attention weights
#         attn_weights = F.softmax(scores, dim=-1)        # [seq_length, seq_length]
        attn_weights = softmax(scores, dim=-1)
        
        # Multiply by values to get output
        output = torch.matmul(attn_weights, V)          # [seq_length, embed_dim]
        return output
