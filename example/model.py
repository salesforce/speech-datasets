import math

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


def get_attn_mask(seq_lens, device=torch.device("cpu")):
    batchsize, maxlen = len(seq_lens), max(seq_lens)
    seq_range = torch.arange(0, maxlen, dtype=torch.int64).unsqueeze(0)
    seq_range_lens = seq_range.new_tensor(seq_lens).unsqueeze(-1)
    return (seq_range < seq_range_lens).to(dtype=torch.float, device=device)


class PositionalEncoding(torch.nn.Module):
    """Positional encoding.

    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length

    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)

        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Conv2dSubsample(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate):
        super().__init__()
        # Convert (N, 1, T, in_dim) to (N, out_dim, T_new, in_dim_new), where
        # T_new = ((T - 1) // 2 - 1) // 2
        # in_dim_new = ((in_dim - 1) // 2 - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_dim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, 3, 2),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            torch.nn.Linear(out_dim * (((in_dim - 1) // 2 - 1) // 2), out_dim),
            PositionalEncoding(out_dim, dropout_rate)
        )

    def forward(self, x, x_mask):
        x = x.unsqueeze(1)  # (N, T, in_dim) --> (N, 1, T, in_dim)
        x = self.conv(x)
        n, c, t, d = x.shape
        x = self.out(x.transpose(1, 2).contiguous().view(n, t, c*d))
        x_mask = x_mask[:, :-2:2][:, :-2:2]
        return x, x_mask


class EncoderDecoder(nn.Module):
    def __init__(self, n_enc_layers, n_dec_layers, input_dim, output_dim,
                 attn_dim=256, dropout_rate=0.1):
        super().__init__()
        assert attn_dim % 64 == 0
        self.conv = Conv2dSubsample(input_dim, attn_dim, dropout_rate)
        enc_config = BertConfig(num_hidden_layers=n_enc_layers,
                                hidden_size=attn_dim,
                                intermediate_size=attn_dim*4,
                                num_attention_heads=attn_dim//64,
                                hidden_dropout_prob=dropout_rate,
                                attention_probs_dropout_prob=dropout_rate,
                                is_decoder=False,
                                max_position_embeddings=1024)
        self.encoder = BertModel(enc_config)

        dec_config = BertConfig(vocab_size=output_dim,
                                num_hidden_layers=n_dec_layers,
                                hidden_size=attn_dim,
                                intermediate_size=attn_dim*4,
                                num_attention_heads=attn_dim//64,
                                hidden_dropout_prob=dropout_rate,
                                attention_probs_dropout_prob=dropout_rate,
                                is_decoder=True,
                                add_cross_attention=True)
        self.decoder = BertModel(dec_config)
        self.out = nn.Linear(attn_dim, output_dim)

    def forward(self, xs, x_lens, ys, y_lens):
        x_mask = get_attn_mask(x_lens, device=xs.device)
        x, x_mask = self.conv(xs, x_mask)
        enc_out = self.encoder(inputs_embeds=x, attention_mask=x_mask)[0]
        y_mask = get_attn_mask(y_lens, device=ys.device)
        dec_out = self.decoder(input_ids=ys.long(), attention_mask=y_mask,
                               encoder_hidden_states=enc_out,
                               encoder_attention_mask=x_mask)[0]
        return self.out(dec_out)

