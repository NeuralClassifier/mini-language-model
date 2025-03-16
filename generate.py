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
    parser.add_argument("--prompt", type=int, required=True, help="Insert the prompt")

   
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if os.path.exists('./model/'):
        
        model_config = torch.load("./model/trained_model.pth")
        
        vocab_size = model_config["vocab_size"]
        embed_dim = model_config["embed_dim"]
        hidden_dim = model_config["hidden_dim"]
        seq_length = model_config["seq_length"]

        model = MiniLM(vocab_size, embed_dim, hidden_dim, seq_length)

        # Load the state dictionary
        model.load_state_dict(model_config["state_dict"])
        model.eval()  # Set to evaluation mode



        seed_text = "hello world hello language model"
        seed_text = args.prompt
        generated_text = generate_text(model, seed_text, generate_len=20, vocab=vocab, seq_length=seq_length)
        print("Generated text:", generated_text)