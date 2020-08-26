import torch.nn as nn

from Layers import Embedding
from LearnedPositionalEmbedding import LearnedPositionalEmbedding
from TransformerLayer import TransformerDecoderLayer, TransformerEncoderLayer


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_vocab_size,
        embedding_dim,
        n_layers,
        hidden_dim,
        n_heads,
        dropout_rate,
        pad_idx,
    ):
        """
        The transformer encoder, functionally a wrapper for embeddings and TransformerEncoderLayers
        Args:
            input_vocab_size:the total length of the input vocab, including special tokens
            embedding_dim:d_model of the transformer
            n_layers: number of layers
            hidden_dim: ffn dimension
            n_heads: number of multihead attn heads
            dropout_rate: overall dropout rate
            pad_idx: padding index
        """

        super(TransformerEncoder, self).__init__()

        self.vocab_size = input_vocab_size
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers

        self.sos_idx = 2
        self.eos_ixd = 3
        self.mask_idx = 0
        self.pad_idx = 1

        self.embedding = Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=pad_idx,
        )

        self.pos_embedding = LearnedPositionalEmbedding(self.embedding_dim)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    self.embedding_dim, hidden_dim, n_heads, dropout_rate
                )
                for layer in range(self.n_layers)
            ]
        )

    def forward(self, x, mask=None):
        """
        Pass input through decoder.
        Args:
            x: raw input indices: [seq_len, bs]
            mask: key padding mask

        Returns:
            Transformed data
        """
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

    def embed(self, source):
        x = self.embedding(source)
        positional_embedding = self.pos_embedding(x)
        x += positional_embedding
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        output_vocab_size,
        embedding_dim,
        n_layers,
        hidden_dim,
        n_heads,
        dropout_rate,
        pad_idx,
    ):
        """
        The transformer decoder, functionally a wrapper for embeddings and TransformerDecoderLayers
        Args:
            output_vocab_size:the total length of the output vocab, including special tokens
            embedding_dim:d_model of the transformer
            n_layers: number of layers
            hidden_dim: ffn dimension
            n_heads: number of multihead attn heads
            dropout_rate: overall dropout rate
            pad_idx: padding index
        """

        super(TransformerDecoder, self).__init__()

        self.vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers

        self.embedding = Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=pad_idx,
        )

        self.pos_embedding = LearnedPositionalEmbedding(self.embedding_dim)

        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    self.embedding_dim, hidden_dim, n_heads, dropout_rate
                )
                for layer in range(self.n_layers)
            ]
        )

    def forward(self, x, memory, source_mask=None, attention_mask=None):
        """
        Pass input through decoder.
        Args:
            x: raw target indices: [seq_len, bs]
            memory: encoder memory: [seq_len, bs, d_model
            source_mask: key padding mask
            attention_mask: attn mask for dec-dec attn

        Returns:
            Transformed data
        """
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x, memory, source_mask, attention_mask)
        return x

    def embed(self, source):
        x = self.embedding(source)
        positional_embedding = self.pos_embedding(x)
        x += positional_embedding
        return x
