import torch
import torch.nn as nn
from vocab_mapping.vocab_mapping import vocabulary_mapping
from backbone_nn.linear.lin import Linear
from backbone_nn.softmax.softm import softmax
from backbone_nn.feed_forward.ff import FeedForward
import numpy as np

# Transformer block combining attention and feed-forward networkj with residuals and layer normalization
class Transformer(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(Transformer, self).__init__()
        self.attention = SelfAttention(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # Self-attention with residual connection and layer norm
        attn_out = self.attention(x)
        x = self.ln1(x + attn_out)
        
        # Feed-forward network with residual connection and layer norm
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        return x