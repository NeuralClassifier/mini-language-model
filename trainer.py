import torch
import torch.nn as nn
# import torch.nn.functional as F
from vocab_mapping.vocab_mapping import vocabulary_mapping
from min_lm.lm import MiniLM
from self_attention.self_attention import SelfAttention
from transformer.transformer import Transformer
from backbone_nn.embeddings.embed import Embedding
from softmax.softm import softmax
import numpy as np
import argparse


def trainer(model, inputs, targets, criterion, optimizer, epochs=100):

    # For simplicity, treat each token prediction as an independent training sample.
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass: model returns logits for each token in the sequence.
        logits = model(inputs)  # shape: [seq_length, vocab_size]
        
        # Compute loss: targets shape is [seq_length] (each target is the next token index)
        loss = criterion(logits, targets)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    

# def tester(model, inputs):
#     # Test: Predict the next token for the last token in our sequence.
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        probs = softmax(logits, dim=-1)
        predicted_indices = torch.argmax(probs, dim=-1)
        print("Input indices:", inputs.tolist())
        print("Predicted next indices:", predicted_indices.tolist())

    return model

