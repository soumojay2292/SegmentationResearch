import torch
import torch.nn as nn
import torch.fft


# --- Mixed Multi-scale Perception Adaptor ---
class MMPA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.proj = nn.Linear(in_channels, 32)
        self.perception1 = nn.Conv2d(32, 32, kernel_size=3, dilation=3, groups=32)
        self.perception2 = nn.Conv2d(32, 32, kernel_size=3, dilation=5, groups=32)
        self.perception3 = nn.Conv2d(32, 32, kernel_size=3, dilation=7, groups=32)
        self.fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        z = self.proj(x)
        e1 = self.perception1(z)
        e2 = self.perception2(z)
        e3 = self.perception3(z)
        weights = self.fc(z.mean(dim=(2,3)))
        out = (weights[:,0].unsqueeze(1)*e1 +
               weights[:,1].unsqueeze(1)*e2 +
               weights[:,2].unsqueeze(1)*e3)
        return out

# --- Frequency Guided Feature Fusion ---
class FGFF(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4)

    def forward(self, spatial, freq):
        B, C, H, W = spatial.shape
        spatial_flat = spatial.view(B, C, -1).permute(2,0,1)
        freq_flat = freq.view(B, C, -1).permute(2,0,1)
        fused, _ = self.attn(spatial_flat, freq_flat, freq_flat)
        return fused.permute(1,2,0).view(B, C, H, W)

# --- Enhanced Decoder Block ---
class EDB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2)
        )
    def forward(self, x):
        return self.block(x)

# --- Reconstruction Decoder Block ---
class RDB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.reconstruct = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1)
    def forward(self, x, original):
        recon = self.reconstruct(x)
        # auxiliary reconstruction loss will be added in training loop
        return recon

# --- MAFFNet ---
class MAFFNet(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.mmpa1 = MMPA(144)
        self.mmpa2 = MMPA(288)
        self.mmpa3 = MMPA(576)
        self.mmpa4 = MMPA(1152)
        self.fgff = FGFF(32)
        self.edb1 = EDB(32, 64)
        self.edb2 = EDB(64, 128)
        self.rdb = RDB(128)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        f1 = self.mmpa1(x1)
        f2 = self.mmpa2(x2)
        f3 = self.mmpa3(x3)
        f4 = self.mmpa4(x4)
        freq = torch.fft.fft2(x)
        freq_mag = torch.abs(freq)
        fused = self.fgff(f4, freq_mag)
        d1 = self.edb1(fused)
        d2 = self.edb2(d1)
        recon = self.rdb(d2, x)
        return d2, recon
