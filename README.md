# mini-language-model

This repository has a simple pipeline for a mini language model, as well as some basic notes about the fundamental concepts of how a language model works which are collected across various sources. The main file, `main.py`, outlines the following steps:

## Overview

```
1. **Text Preprocessing**: Convert a sample text into tokenized inputs and targets using a vocabulary mapping.
2. **Embedding Layer**: Map token indices to dense embedding vectors.
3. **Positional Encoding**: Add learnable positional information to the embeddings.
4. **Self-Attention**: Compute self-attention over the sequence.
5. **Transformer Block**: Apply a transformer block to further process the sequence.
6. **MiniLM Model**: Combine the above components into a final language model architecture.
```

## Detailed Step-by-Step Explanation

### 1. Import Dependencies and Define Argument Parser

- **Imports**: The script imports PyTorch modules along with custom modules:
  - `vocabulary_mapping` from `vocab_mapping.vocab_mapping`: Converts text into numeric tokens.
  - `Embedding` from `backbone_nn.embeddings.embed`: Lookup table for embeddings.
  - `SelfAttention` from `self_attention.self_attention`: Implements self-attention mechanism.
  - `Transformer` from `transformer.transformer`: A transformer block for further processing.
  - `MiniLM` from `min_lm.lm`: The final language model that combines all components.
- **Argument Parsing**: Uses `argparse` to require two parameters:
  - `--embed_dim`: The size of the embedding vectors.
  - `--hidden_dim`: The hidden dimension size used in the transformer block.

### 2. Main Execution Block

- **Sample Text**: A default text is defined:
  ```python
  text = "hello world hello language model hello deep learning hello AI"

