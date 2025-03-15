import torch
import torch.nn as nn
# import torch.nn.functional as F
from vocab_mapping.vocab_mapping import vocabulary_mapping
from backbone_nn.embeddings.embed import Embedding
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Mini LLM")
    parser.add_argument("--embed_dim", type=int, required=True, help="Select embedding layer size")
   
    return parser.parse_args()

if __name__ == "__main__":
    # Sample text (you can replace this with any text data)
    text = "hello world hello language model hello deep learning hello AI"

    # with open('/Users/kushankurghosh/Documents/fictionStorydata.txt') as file:
    #     text = file.read()

    inputs, targets = vocabulary_mapping(text)

    args = parse_args()
    embed_dim = args.embed_dim # dimension of embedding vector

    # embedding layer
    embedding_layer = Embedding(vocab_size, embed_dim)

    # Lookup embeddings for our inputs
    embedded = embedding_layer(inputs)  # shape: [sequence_length, embed_dim]
    print("Embedded shape:", embedded.shape)

    # Create a learnable positional encoding for our sequence length.
    seq_length = inputs.shape[0]
    positional_encoding = nn.Parameter(torch.zeros(seq_length, embed_dim))

    # Adding positional encoding to embedded tokens
    x = embedded + positional_encoding  # shape: [seq_length, embed_dim]

    # Instantiating and testing self-attention on input x.
    self_attention = SelfAttention(embed_dim)
    attn_output = self_attention(x)
    print("Attention output shape:", attn_output.shape)


