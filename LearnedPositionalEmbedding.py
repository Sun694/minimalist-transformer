import torch
import torch.nn as nn

from utils import device


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        """
        Learned positional encoding function.
        Args:
            d_model: d_model of transformer
            max_len: the absolute maximum expected length of a given sample
        """
        super(LearnedPositionalEmbedding, self).__init__()
        pe = nn.Embedding(num_embeddings=max_len, embedding_dim=d_model,)
        self.positional_encoding = pe

    def forward(self, x):
        x = x + self.positional_encoding(torch.arange(x.size(0)).to(device)).unsqueeze(
            1
        )
        return x
