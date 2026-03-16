import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic convolutional block
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

# Transformer encoder block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (seq_len, batch, embed_dim)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x

# TransUNet model
class TransUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, embed_dim=128, num_heads=4):
        super().__init__()
        # Encoder (CNN)
        self.enc1 = ConvBlock(in_ch, 64)
        self.enc2 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)

        # Transformer
        self.flatten = nn.Flatten(2)  # flatten H,W into sequence
        self.proj = nn.Linear(128, embed_dim)
        self.transformer = TransformerBlock(embed_dim, num_heads=num_heads)

        # Decoder (CNN)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = ConvBlock(embed_dim, 64)
        self.final = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        # CNN encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))

        # Transformer
        b, c, h, w = x2.shape
        seq = x2.flatten(2).permute(2, 0, 1)  # (seq_len, batch, channels)
        seq = self.proj(seq)
        seq = self.transformer(seq)
        x_t = seq.permute(1, 2, 0).reshape(b, -1, h, w)

        # Decoder
        x_up = self.up(x_t)
        x_out = self.dec1(x_up)
        return torch.sigmoid(self.final(x_out))
