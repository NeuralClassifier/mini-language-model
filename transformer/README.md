# Transformer

This code is a simple Transformer model with self-attention, layer normalization, and feed-forward networks.

## Overview

It consists of:

1. **Self-Attention**: Computes attention scores for each token in a sequence.
2. **Residual Connections**: Helps gradients flow and prevents vanishing gradients.
3. **Layer Normalization**: Normalizes activations for stability.
4. **Feed-Forward Network (FFN)**: Applies a multi-layer perceptron for feature transformation.

## Mathematical Formulation

Given an input tensor \( X \), the forward pass follows:

1. **Self-Attention with Residual Connection**
   \[
   A = \text{SelfAttention}(X)
   \]
   \[
   X' = \text{LayerNorm}(X + A)
   \]

2. **Feed-Forward Network with Residual Connection**
   \[
   F = \text{FFN}(X')
   \]
   \[
   Y = \text{LayerNorm}(X' + F)
   \]

where:
- `SelfAttention` computes attention scores.
- `FFN` applies a transformation.
- Residual connections help preserve information.
- Layer normalization stabilizes training.

## Usage

```python
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_weights = self.softmax(q @ k.transpose(-2, -1) / (x.shape[-1] ** 0.5))
        return attn_weights @ v

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class Transformer(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(Transformer, self).__init__()
        self.attention = SelfAttention(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out = self.attention(x)
        x = self.ln1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        return x
