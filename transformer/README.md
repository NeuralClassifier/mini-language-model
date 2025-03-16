# Transformer

This code is a simple Transformer model with self-attention, layer normalization, and feed-forward networks.

## Overview

It consists of:

1. **Self-Attention**: Computes attention scores for each token in a sequence.
2. **Residual Connections**: Helps gradients flow and prevents vanishing gradients.
3. **Layer Normalization**: Normalizes activations for stability.
4. **Feed-Forward Network (FFN)**: Applies a multi-layer perceptron for feature transformation.

## Mathematical Formulation

Given an input tensor $X$, the forward pass follows:

1. **Self-Attention with Residual Connection**
   $$
   A = SelfAttention(X)
   $$
   $$
   X' = LayerNorm(X + A)
   $$

2. **Feed-Forward Network with Residual Connection**
   $$
   F = FFN(X')
   $$
   $$
   Y = LayerNorm(X' + F)
   $$

where:
- `SelfAttention` computes attention scores.
- `FFN` applies a transformation.
- Residual connections help preserve information.
- Layer normalization stabilizes training.
