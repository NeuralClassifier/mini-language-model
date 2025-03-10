import torch

def vocabulary_mapping(text):

    tokens = text.split()
    vocab = {token: i for i, token in enumerate(set(tokens))}
    vocab_size = len(vocab)

    data = [vocab[token] for token in tokens]

    return torch.tensor(data[:-1], dtype=torch.long), torch.tensor(data[1:], dtype=torch.long)