import torch.nn as nn


def Embedding(num_embeddings, embedding_dim, padding_idx):
    """
    Simple wrapper for embedding layers that handles initialization
    Args:
        num_embeddings: vocab size
        embedding_dim: embedding dimension
        padding_idx: padding index

    Returns:
        torch.nn.Embedding properly initialized
    """
    m = nn.Embedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        padding_idx=padding_idx,
    )
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    """
    Simple wrapper for Linear layers that handles initialization
    Args:
        in_features: input dimension
        out_features: output dimension
        bias: bool, whether or not to add bias

    Returns:
        torch.nn.Linear properly initialized
    """
    m = nn.Linear(in_features, out_features, bias)
    nn.init.kaiming_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0)
    return m
