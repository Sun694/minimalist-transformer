import torch.nn as nn
import torch.nn.functional as F

# from MultiheadAttention import MultiheadAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_heads, dropout_rate=0):
        """
        Single transformer encoder layer.
        Args:
            embedding_dim: d_model of transformer
            hidden_dim: ffn dimension
            n_heads: number of multihead attn heads
            dropout_rate: overall dropout rate

            uses prenorm, not post-norm
        """
        super(TransformerEncoderLayer, self).__init__()
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        self.self_attention = nn.MultiheadAttention(self.embedding_dim, self.n_heads)
        self.fc1 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.embedding_dim)

        self.self_attention_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x, source_mask=None):
        """
        Pass input through single encoder layer.
        Args:
            x: embedded source: [seq_len, bs, d_model]
            source_mask: key padding mask

        Returns:
            input after a single transformer encoder layer
        """
        r = x
        x = self.self_attention_layer_norm(x)
        x, _ = self.self_attention(
            query=x, key=x, value=x, key_padding_mask=source_mask
        )
        x = x + r

        r = x
        x = self.final_layer_norm(x)
        x = self.fc2(self.dropout(F.relu(self.fc1(x.transpose(0, 1))))).transpose(0, 1)
        x = self.dropout(x)
        x = x + r

        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_heads, dropout_rate=0):
        """
        Single decoder layer.
        Args:
            embedding_dim: d_model of transformer
            hidden_dim: ffn dimension
            n_heads: number of multihead attn heads
            dropout_rate: overall dropout rate

            uses prenorm, not post-norm
        """
        super(TransformerDecoderLayer, self).__init__()
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim

        self.self_attention = nn.MultiheadAttention(self.embedding_dim, self.n_heads,)
        self.self_attention_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.enc_attention = nn.MultiheadAttention(self.embedding_dim, self.n_heads,)
        self.enc_attention_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.fc1 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.embedding_dim)

        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x, enc_out, source_mask=None, tgt_mask=None):
        """
        Passes input through single decoder layer.
        Args:
            x: embedded targets: [seq_len, bs, d_model]
            enc_out: encoder memory
            source_mask: key padding mask
            tgt_mask: attn mask for dec-dec attn

        Returns:
            input after a single transformer decoder layer
        """
        r = x
        x = self.self_attention_layer_norm(x)
        x, _ = self.self_attention(query=x, key=x, value=x, attn_mask=tgt_mask)
        x = self.dropout(x)
        x = x + r

        r = x
        x = self.enc_attention_layer_norm(x)
        x, _ = self.enc_attention(
            query=x, key=enc_out, value=enc_out, key_padding_mask=source_mask
        )
        x = self.dropout(x)
        x = x + r

        r = x
        x = self.final_layer_norm(x)
        x = self.fc2(self.dropout(F.relu(self.fc1(x.transpose(0, 1))))).transpose(0, 1)
        x = self.dropout(x)
        x = x + r

        return x
