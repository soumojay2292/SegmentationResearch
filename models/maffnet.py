import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# MMPA (Paper-aligned simplified)
# ------------------------------
class MMPA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.proj = nn.Conv2d(in_channels, 32, 1)

        self.e1 = nn.Conv2d(32, 32, 3, padding=3, dilation=3, groups=32)
        self.e2 = nn.Conv2d(32, 32, 3, padding=5, dilation=5, groups=32)
        self.e3 = nn.Conv2d(32, 32, 3, padding=7, dilation=7, groups=32)

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 3),
            nn.Softmax(dim=-1)
        )

        self.reproj = nn.Conv2d(32, in_channels, 1)

    def forward(self, x):
        z = self.proj(x)

        e1 = self.e1(z)
        e2 = self.e2(z)
        e3 = self.e3(z)

        weights = self.fc(z)

        w1 = weights[:, 0].view(-1, 1, 1, 1)
        w2 = weights[:, 1].view(-1, 1, 1, 1)
        w3 = weights[:, 2].view(-1, 1, 1, 1)

        fused = w1 * e1 + w2 * e2 + w3 * e3

        return self.reproj(fused + z)


# ------------------------------
# Frequency Encoder (Paper aligned)
# ------------------------------
class FrequencyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1)

    def forward(self, x):
        # grayscale
        x_gray = x.mean(dim=1, keepdim=True)

        # FFT
        freq = torch.fft.fft2(x_gray.float())
        mag = torch.abs(freq)

        # stabilization
        mag = torch.log1p(mag)
        mag = mag / (mag.mean(dim=(2, 3), keepdim=True) + 1e-6)
        mag = torch.nan_to_num(mag)

        # replicate channels
        mag = mag.repeat(1, 3, 1, 1)

        return self.conv(mag)


# ------------------------------
# FGFF (Paper-style attention fusion)
# ------------------------------
class FGFF(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.spatial_conv = nn.Conv2d(channels, channels, 3, padding=1)

        self.attn = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, spatial, freq):

        spatial_feat = self.spatial_conv(spatial)

        avg = torch.mean(freq, dim=1, keepdim=True)
        mx, _ = torch.max(freq, dim=1, keepdim=True)

        attn = self.attn(torch.cat([avg, mx], dim=1))

        return spatial_feat * attn


# ------------------------------
# Decoder Block
# ------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.conv(x)


# ------------------------------
# MAFFNet FINAL
# ------------------------------
class MAFFNet(nn.Module):
    def __init__(self, encoder):
        super().__init__()

        self.encoder = encoder

        # freeze SAM
        for p in self.encoder.parameters():
            p.requires_grad = False

        # channel alignment (SAM → 256 assumed)
        self.channel_proj = nn.Conv2d(256, 64, 1)

        # modules
        self.mmpa = MMPA(64)
        self.freq_enc = FrequencyEncoder()
        self.fgff = FGFF(64)

        # decoder
        self.dec1 = DecoderBlock(64, 64)
        self.dec2 = DecoderBlock(64, 32)
        self.dec3 = DecoderBlock(32, 16)

        self.final_conv = nn.Conv2d(16, 1, 1)

    def forward(self, x):

        # --------------------------
        # STEP 1: SAM input
        # --------------------------
        x_sam = F.interpolate(x, (1024, 1024), mode='bilinear', align_corners=False)

        with torch.no_grad():
            features = self.encoder.image_encoder(x_sam)

        # --------------------------
        # STEP 2: resize back
        # --------------------------
        features = F.interpolate(features, (224, 224), mode='bilinear', align_corners=False)

        x_spatial = self.channel_proj(features)

        # --------------------------
        # STEP 3: MMPA
        # --------------------------
        x_spatial = self.mmpa(x_spatial)

        # --------------------------
        # STEP 4: Frequency branch
        # --------------------------
        x_freq = self.freq_enc(x)

        x_freq = F.interpolate(x_freq, size=x_spatial.shape[-2:], mode='bilinear')

        # --------------------------
        # STEP 5: FGFF fusion
        # --------------------------
        x_fused = self.fgff(x_spatial, x_freq)

        # --------------------------
        # STEP 6: Decoder
        # --------------------------
        d1 = self.dec1(x_fused)
        d2 = self.dec2(d1)
        d3 = self.dec3(d2)

        out = self.final_conv(d3)

        # --------------------------
        # FINAL OUTPUT
        # --------------------------
        out = F.interpolate(out, size=(224, 224), mode='bilinear')

        return out