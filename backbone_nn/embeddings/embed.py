import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Embedding, self).__init__()
        self.embedding_matrix = nn.Parameter(torch.randn(vocab_size, embed_dim))  # Randomly initialized weights

    def forward(self, input_indices):
        return self.embedding_matrix[input_indices]  # 