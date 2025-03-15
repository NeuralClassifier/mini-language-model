import torch
import torch.nn as nn
import sys
import o

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

from backbone_nn.linear.lin import Linear
from backbone_nn.softmax.softm import softmax
from backbone_nn.feed_forward.ff import FeedForward

from transformer.transformer import Transformer
from backbone_nn.embeddings.embed import Embedding
import numpy as np

class MiniLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, seq_length):
        super(MiniLM, self).__init__()
        self.embedding = Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(seq_length, embed_dim))
        self.transformer = Transformer(embed_dim, hidden_dim)
        self.output_proj = Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        # x shape: [seq_length]
        emb = self.embedding(x)  # [seq_length, embed_dim]
        emb = emb + self.positional_encoding  # add position info
        transformer_out = self.transformer(emb)  # [seq_length, embed_dim]
        logits = self.output_proj(transformer_out)  # [seq_length, vocab_size]
        return logits