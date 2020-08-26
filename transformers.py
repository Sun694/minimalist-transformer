import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from EncoderDecoder import TransformerEncoder, TransformerDecoder
from Layers import Linear
from utils import Batch, device


class BaseTransformer(nn.Module):
    def __init__(
        self,
        input_vocab_size,
        output_vocab_size,
        embedding_dim,
        n_layers,
        hidden_dim,
        n_heads,
        dropout_rate,
    ):
        """
        Boilerplate encoder decoder transformer. Does not implement forward_and_return_loss or generation.
        Args:
            input_vocab_size: the total length of the input vocab, including special tokens
            output_vocab_size: the total length of the output vocab, including special tokens
            embedding_dim: d_model of the transformer
            n_layers: number of encoder/decoder layers in the model
            hidden_dim: feedforward dimension of post-attn FFNs
            n_heads: number of heads per multihead attention
            dropout_rate: global dropout rate
        """

        super(BaseTransformer, self).__init__()
        self.output_vocab_size = output_vocab_size
        self.input_vocab_size = input_vocab_size
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.sos_idx = 2
        self.eos_ixd = 3
        self.mask_idx = 0
        self.pad_idx = 1

        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.embedding_scale = np.sqrt(self.embedding_dim)

        self.fc1 = Linear(self.embedding_dim, self.output_vocab_size)

        self.encoder = TransformerEncoder(
            self.input_vocab_size,
            self.embedding_dim,
            self.n_layers,
            self.hidden_dim,
            self.n_heads,
            self.dropout_rate,
            self.pad_idx,
        )

        self.decoder = TransformerDecoder(
            self.input_vocab_size,
            self.embedding_dim,
            self.n_layers,
            self.hidden_dim,
            self.n_heads,
            self.dropout_rate,
            self.pad_idx,
        )

    def forward(self, source, targets, source_mask=None, tgt_mask=None):
        """
        Pass the input through the transformer
        Args:
            source: source sequence indices, [seq_len, bs]
            targets: target sequence shifted right indices, [full_seq_len - 1, bs]
            source_mask: key padding mask for enc self attn and dec-enc attn
            tgt_mask: attn mask for dec-dec attn

        Returns:
            output after being passed through transformer, [full_seq_len - 1, bs, out_vocab_size]
        """

        y = self.decoder(
            targets, self.encoder(source, source_mask), source_mask, tgt_mask
        )
        y = self.fc1(y.transpose(0, 1)).transpose(0, 1)
        return y

    def forward_and_return_loss(self, *args):
        raise NotImplementedError

    def generate(self, *args):
        raise NotImplementedError


class MLT(BaseTransformer):
    def __init__(
        self, *args,
    ):
        super(MLT, self).__init__(*args)

    def forward_and_return_loss(self, criterion, sources, targets):
        """
        Pass input through transformer and return loss, handles masking automagically
        Args:
            criterion: torch.nn.functional loss function of choice
            sources: source sequences, [seq_len, bs, indices]
            targets: full target sequence, [seq_len, bs, embedding_dim]

        Returns:
            loss, transformer output
        """

        batch = Batch(sources, targets, self.pad_idx)
        seq_len, batch_size = batch.trg.size()
        out = self.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = criterion(
            out.contiguous().view(-1, out.size(-1)),
            batch.trg_y.contiguous().view(-1),
            ignore_index=self.pad_idx,
        )

        return loss, out

    def generate(self, source, source_mask, max_len):
        memory = self.encoder(source, source_mask)
        ys = torch.ones(1, source.size(1)).long().fill_(self.sos_idx).to(device)
        # max target length is 1.5x * source + 10 to save compute power
        for _ in range(int(1.5 * source.size(0)) - 1 + 10):
            out = self.decoder(ys, memory, source_mask, Batch(ys, ys, 1).raw_mask)
            out = self.fc1(out[-1].unsqueeze(0))
            prob = F.log_softmax(out, dim=-1)
            next_word = torch.argmax(prob, dim=-1)
            ys = torch.cat([ys, next_word.detach()], dim=0)

        return ys
