import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Positional Encoding ----------------
class PositionalEncoding2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        if C != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {C}")

        pe = torch.zeros(C, H, W, device=device)
        y_pos = torch.arange(H, device=device).float().unsqueeze(1)  # [H,1]
        x_pos = torch.arange(W, device=device).float().unsqueeze(0)  # [1,W]

        div_term = torch.exp(
            torch.arange(0, C // 2, device=device).float()
            * -(torch.log(torch.tensor(10000.0, device=device)) / (C // 2))
        )

        # ✅ Correct broadcasting
        pe[0::2, :, :] = torch.sin(y_pos * div_term.unsqueeze(0)).unsqueeze(2).repeat(1, 1, W)
        pe[1::2, :, :] = torch.cos(x_pos * div_term.unsqueeze(1)).unsqueeze(0).repeat(H, 1, 1)

        return x + pe.unsqueeze(0)

# ---------------- TransUNet ----------------
class TransUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, embed_dim=128, num_heads=4):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, embed_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # Positional encoding
        self.pos_enc = PositionalEncoding2D(embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Decoder
        self.up1 = nn.ConvTranspose2d(embed_dim, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)        # [B,64,H,W]
        x2 = self.enc2(x1)       # [B,embed_dim,H/2,W/2]

        # Positional encoding
        x2 = self.pos_enc(x2)

        # Flatten for transformer
        B, C, H, W = x2.shape
        x2_flat = x2.flatten(2).permute(2, 0, 1)  # [HW,B,C]
        x2_trans = self.transformer(x2_flat)
        x2 = x2_trans.permute(1, 2, 0).view(B, C, H, W)

        # Decoder with skip connection
        x_up = self.up1(x2)      # [B,64,H,W]
        x_cat = torch.cat([x_up, x1], dim=1)
        x_out = self.dec1(x_cat)
        return self.out_conv(x_out)
