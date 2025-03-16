import torch
import torch.nn as nn
from vocab_mapping.vocab_mapping import vocabulary_mapping
from min_lm.lm import MiniLM
from self_attention.self_attention import SelfAttention
from transformer.transformer import Transformer
from backbone_nn.embeddings.embed import Embedding
from trainer import trainer
from softmax.softm import softmax
import numpy as np
import argparse



def generate_text(model, seed_text, generate_len, vocab, seq_length):
    """
    Generates text using a sliding window approach.
    
    Args:
      model: The trained language model.
      seed_text: (Prompt).
      generate_len: Number of tokens to generate.
      vocab: Dictionary mapping tokens to indices.
      seq_length: Fixed sequence length the model expects.
      
    Returns:
      A string with the generated text.
    """
    # Tokenize the seed text.
    seed_tokens = seed_text.split()
    # Convert tokens to indices.
    seed_indices = [vocab.get(token, 0) for token in seed_tokens]  # use index 0 if token not found

    inv_vocab = {i: token for token, i in vocab.items()}
    
    # Ensure the seed has exactly `seq_length` tokens:
    if len(seed_indices) < seq_length:
        # If shorter, repeat the seed until reaching the required length.
        while len(seed_indices) < seq_length:
            seed_indices.extend(seed_indices)
        seed_indices = seed_indices[:seq_length]
    elif len(seed_indices) > seq_length:
        # If longer, take only the last seq_length tokens.
        seed_indices = seed_indices[-seq_length:]
    
    # Convert the list into a tensor.
    current_seq = torch.tensor(seed_indices, dtype=torch.long)
    model.eval()  # set model to evaluation mode
    generated_tokens = seed_tokens.copy()
    
    # Generation loop.
    for _ in range(generate_len):
        with torch.no_grad():
            logits = model(current_seq)  # shape: [seq_length, vocab_size]
            # We use the logits corresponding to the last token in the window.
            last_logits = logits[-1]
            # Convert logits to probabilities.
            probs = softmax(last_logits, dim=0)
            # Sample the next token from the probability distribution.
            next_token_idx = torch.multinomial(probs, num_samples=1).item()
            
            # Append the generated token (convert index back to token).
            generated_tokens.append(inv_vocab[next_token_idx])
            
            # Update current_seq: slide the window by dropping the first token and appending the new token.
            current_seq = torch.cat([current_seq[1:], torch.tensor([next_token_idx], dtype=torch.long)])
    
    return ' '.join(generated_tokens)