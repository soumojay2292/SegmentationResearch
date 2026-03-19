import torch, torch.nn as nn
import torch.fft

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
        y_pos = torch.arange(0, H, device=device).unsqueeze(1).float()
        x_pos = torch.arange(0, W, device=device).unsqueeze(0).float()

        div_term = torch.exp(
            torch.arange(0, C // 2, device=device).float() * -(torch.log(torch.tensor(10000.0, device=device)) / (C // 2))
        )

        pe[0::2, :, :] = torch.sin(y_pos * div_term.unsqueeze(0)).unsqueeze(2).repeat(1, 1, W)
        pe[1::2, :, :] = torch.cos(x_pos * div_term.unsqueeze(1)).unsqueeze(0).repeat(H, 1, 1)

        return x + pe.unsqueeze(0)



class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).permute(2, 0, 1)   # [HW, B, C]
        attn, _ = self.attn(seq, seq, seq)
        seq = seq + attn
        seq = seq + self.ff(seq)
        out = seq.permute(1, 2, 0).contiguous().view(B, C, H, W)
        return out


class MAFFNet(nn.Module):
    def __init__(self, height=128, width=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU()
        )
        self.pos_enc = PositionalEncoding2D(128)
        self.trans = TransformerBlock(128)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        feat = self.enc(x)
        freq = torch.fft.fft2(feat.float())
        freq = torch.abs(freq)
        fused = feat + freq
        fused = self.pos_enc(fused)   # ✅ add positional encoding
        fused = self.trans(fused)     # ✅ transformer block
        return torch.sigmoid(self.dec(fused))
