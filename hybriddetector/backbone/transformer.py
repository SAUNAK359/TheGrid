# hybriddetector/backbone/transformer.py

import torch
import torch.nn as nn
import math

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels, height, width):
        super(PositionalEncoding2D, self).__init__()
        if channels % 4 != 0:
            raise ValueError("Channels must be divisible by 4 for 2D PE")
        self.channels = channels
        self.height = height
        self.width = width
        self.register_buffer('pos_encoding', self._build_pe())

    def _build_pe(self):
        C, H, W = self.channels, self.height, self.width
        pe = torch.zeros(C, H, W)
        channels_per_dim = C // 2
        div_term = torch.exp(torch.arange(0, channels_per_dim, 2) * -(math.log(10000.0) / channels_per_dim))

        y_pos = torch.arange(H).unsqueeze(1)
        x_pos = torch.arange(W).unsqueeze(1)

        # Height
        pe_h = torch.zeros(channels_per_dim, H, 1)
        for idx in range(0, channels_per_dim, 2):
            pe_h[idx, :, 0] = torch.sin(y_pos[:, 0] * div_term[idx // 2])
            if idx + 1 < channels_per_dim:
                pe_h[idx+1, :, 0] = torch.cos(y_pos[:, 0] * div_term[idx // 2])

        # Width
        pe_w = torch.zeros(channels_per_dim, 1, W)
        for idx in range(0, channels_per_dim, 2):
            pe_w[idx, 0, :] = torch.sin(x_pos[:, 0] * div_term[idx // 2])
            if idx + 1 < channels_per_dim:
                pe_w[idx+1, 0, :] = torch.cos(x_pos[:, 0] * div_term[idx // 2])

        pe[:channels_per_dim, :, :] = pe_h.repeat(1, 1, W)
        pe[channels_per_dim:, :, :] = pe_w.repeat(1, H, 1)
        return pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pos_encoding[:, :x.size(1), :x.size(2), :x.size(3)]


class MHSABlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(MHSABlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Linear(embed_dim*4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1,2)
        x_attn, _ = self.attn(x_flat, x_flat, x_flat)
        x = self.norm1(x_flat + self.dropout(x_attn))
        x_ffn = self.ffn(x)
        x = self.norm2(x + self.dropout(x_ffn))
        x = x.transpose(1,2).view(B,C,H,W)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels, height, width, num_heads=8, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.pos_encoding = PositionalEncoding2D(channels, height, width)
        self.mhsa = MHSABlock(embed_dim=channels, num_heads=num_heads, dropout=dropout)

    def forward(self, x):
        x = self.pos_encoding(x)
        x = self.mhsa(x)
        return x


if __name__ == "__main__":
    dummy_input = torch.randn(1, 512, 40, 40)
    model = TransformerBlock(512, 40, 40)
    out = model(dummy_input)
    print(out.shape)
