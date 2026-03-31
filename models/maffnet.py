from pyexpat import features

import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F


# --- Mixed Multi-scale Perception Adaptor ---
class MMPA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.perception1 = nn.Conv2d(32, 32, kernel_size=3, padding=3, dilation=3, groups=32)
        self.perception2 = nn.Conv2d(32, 32, kernel_size=3, padding=5, dilation=5, groups=32)
        self.perception3 = nn.Conv2d(32, 32, kernel_size=3, padding=7, dilation=7, groups=32)
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
        w1 = weights[:,0].view(-1,1,1,1)
        w2 = weights[:,1].view(-1,1,1,1)
        w3 = weights[:,2].view(-1,1,1,1)

        out = w1 * e1 + w2 * e2 + w3 * e3
        return out

# --- Frequency Guided Feature Fusion ---
class FGFF(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 2, in_channels, 1)
        self.gate = nn.Sigmoid()

    def forward(self, spatial, freq):
        fused = torch.cat([spatial, freq], dim=1)
        weight = self.gate(self.conv(fused))
        return spatial * weight + freq * (1 - weight)

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
        self.channel_proj = nn.Conv2d(256, 64, kernel_size=1)
        self.freq_proj = nn.Conv2d(64, 32, kernel_size=1)
        self.final_conv = nn.Conv2d(128, 1, kernel_size=1)

        self.mmpa1 = MMPA(64)
        self.mmpa2 = MMPA(64)
        self.mmpa3 = MMPA(64)
        self.mmpa4 = MMPA(64)
        self.fgff = FGFF(32)
        self.edb1 = EDB(32, 64)
        self.edb2 = EDB(64, 128)
        self.rdb = RDB(128)

    def forward(self, x):
        x_sam = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)
        features = self.encoder.image_encoder(x_sam)
        features = F.interpolate(features, size=(224, 224), mode='bilinear', align_corners=False)
        x4 = self.channel_proj(features)
        x3 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x2 = F.interpolate(x4, scale_factor=4, mode='bilinear', align_corners=False)
        x1 = F.interpolate(x4, scale_factor=8, mode='bilinear', align_corners=False)
        f1 = self.mmpa1(x1)
        f2 = self.mmpa2(x2)
        f3 = self.mmpa3(x3)
        f4 = self.mmpa4(x4)
        x4_float = x4.float()                     # force FP32
        with torch.cuda.amp.autocast(enabled=False):
            freq = torch.fft.fft2(x4.float())
        freq_mag = torch.abs(freq)
        freq_mag = torch.log1p(freq_mag)
        # freq_mag = torch.clamp(freq_mag, 0, 1)
        freq_mag = freq_mag / (freq_mag.amax(dim=(2,3), keepdim=True) + 1e-8)   
        freq_mag = torch.nan_to_num(freq_mag)
        freq_mag = freq_mag.to(x4.dtype)
        freq_mag = self.freq_proj(freq_mag)   # ⭐ THIS LINE FIXES EVERYTHING
        fused = self.fgff(f4, freq_mag)
        d1 = self.edb1(fused)
        d2 = self.edb2(d1)
        out = self.final_conv(d2)
        out = F.interpolate(out, size=(224, 224), mode='bilinear', align_corners=False)
        return out
