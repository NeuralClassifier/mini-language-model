import os
import torch
import torch.nn as nn
from vocab_mapping.vocab_mapping import vocabulary_mapping
from min_lm.lm import MiniLM
from self_attention.self_attention import SelfAttention
from transformer.transformer import Transformer
from backbone_nn.embeddings.embed import Embedding
from trainer import trainer
from generate_text import generate_text
import numpy as np
import argparse
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="Mini LLM")
    parser.add_argument("--embed_dim", type=int, required=True, help="Select embedding layer size")
    parser.add_argument("--hidden_dim", type=int, required=True, help="Select hidde layer size in the transformer")
    parser.add_argument("--lr", type=int, required=True, help="Choose a learning rate")
    parser.add_argument("--epochs", type=int, required=True, help="Total epochs to train the transformer")
    parser.add_argument("--dataset", type=str, required=True, help="Mention subset/full data")
   
    return parser.parse_args()

if __name__ == "__main__":

    if os.path.exists('./model/'):
        shutil.rmtree('./model/')
    
    text = "hello world hello language model hello deep learning hello AI" # Sample text (any user can replace this with any text data)

    args = parse_args()

    if args.dataset.lower() == 'subset':

        with open('./dataset/fictionStorydata_subset.txt') as file:
            text = file.read()
            
    elif args.dataset.lower() == 'full':

        with open('./dataset/fictionStorydata_full.txt') as file:
            text = file.read()

    inputs, targets, vocab_size = vocabulary_mapping(text)

    
    embed_dim = args.embed_dim

    # embedding layer
    embedding_layer = Embedding(vocab_size, embed_dim)

    # lookup embeddings for our inputs
    embedded = embedding_layer(inputs)  # shape: [sequence_length, embed_dim]
    print("Embedded shape:", embedded.shape)

    # Create a learnable positional encoding for our sequence length.
    seq_length = inputs.shape[0]
    positional_encoding = nn.Parameter(torch.zeros(seq_length, embed_dim))

    # adding positional encoding to embedded tokens
    x = embedded + positional_encoding  # shape: [seq_length, embed_dim]

    # esting self-attention on input x.
    self_attention = SelfAttention(embed_dim)
    attn_output = self_attention(x)
    print("Attention output shape:", attn_output.shape)

    # transformer block.
    hidden_dim = args.hidden_dim
    transformer_block = Transformer(embed_dim, hidden_dim)
    transformer_output = transformer_block(x)
    print("Transformer block output shape:", transformer_output.shape)

    model = MiniLM(vocab_size, embed_dim, hidden_dim, seq_length)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    trained_model = trainer(model, inputs, targets, criterion, optimizer, epochs=args.epochs)

    if not os.path.exists('./model/'):
        os.makedirs('./model/')

    torch.save(trained_model.state_dict(), './model/trained_model.pth')


# seed_text = "hello world hello language model"
# generated_text = generate_text(model, seed_text, generate_len=10, vocab=vocab, seq_length=seq_length)
# print("Generated text:", generated_text)