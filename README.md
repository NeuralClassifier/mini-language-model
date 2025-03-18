# mini-language-model

This repository has a simple pipeline for a mini language model, as well as some basic notes about the fundamental concepts of how a language model works which are collected across various sources. The main file, `train_main.py`, outlines the following steps:

## Overview of `train_main.py`

```
1. Text Preprocessing: Converting a sample text into tokenized inputs and targets using a vocabulary mapping dict.
2. Embedding Layer: Mapping token indices to embedding vectors
3. Positional Encoding: Adding learnable positional information to the embeddings
4. Self-Attention: Computing self-attention over the sequence
5. Transformer Block: Applying a transformer pipeline
6. MiniLM Model: Combining the above components into a final model
```
### 1. Argument Parser

**Uses `argparse` to require two parameters**:
  - `--embed_dim`: The size of the embedding vectors.
  - `--hidden_dim`: The hidden dimension size used in the transformer block.
  - `--lr`: The learning rate for the model
  - `--epochs`: Total epochs to train the transformer
  - `--dataset`: To train on full dataset or a subset

### 2. Some Notes to look for:
  - Self-Attention: Click here

### 3. How to train the model?

Change the parameters as per your convinience

```
python train_main.py --embed_dim 16 --hidden_dim 64 --lr 0.01 --epochs 100 --dataset subset
```

### 4. How to use the trained model?

```
python generate.py --prompt *ENTER THE PROMPT HERE*
```
