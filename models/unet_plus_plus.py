"""UNet++ (Zhou et al., 2018) for binary segmentation.

Implements nested dense skip connections across 4 encoder levels.
All nodes X_{i,j} (j>=1) aggregate every prior node at the same depth
plus one bilinearly-upsampled feature from the level below.

Input:  (B, 3, H, W)  — RGB image
Output: (B, 1, H, W)  — raw logits (no sigmoid applied)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two 3×3 conv layers with BN + ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNetPlusPlus(nn.Module):
    """UNet++ with 4 encoder levels and dense nested skip connections.

    Node grid (i=depth level, j=dense column):

        X_{0,0} → X_{0,1} → X_{0,2} → X_{0,3}  ← output
        X_{1,0} → X_{1,1} → X_{1,2}
        X_{2,0} → X_{2,1}
        X_{3,0}                                   ← deepest (bottleneck)

    Each X_{i,j} (j≥1) concatenates:
        [X_{i,0}, …, X_{i,j-1},  upsample(X_{i+1, j-1})]

    Channel sizes per level: [64, 128, 256, 512]
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super().__init__()

        filters = [64, 128, 256, 512]
        depth   = len(filters)  # 4

        # ── Encoder (column j=0) ─────────────────────────────────
        enc_in = [in_ch] + filters[:-1]          # [3, 64, 128, 256]
        self.enc  = nn.ModuleList([
            _conv_block(enc_in[i], filters[i]) for i in range(depth)
        ])
        self.pool = nn.MaxPool2d(2)

        # ── Upsamplers: up[i] maps filters[i+1] → filters[i] ────
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(filters[i + 1], filters[i], kernel_size=2, stride=2)
            for i in range(depth - 1)
        ])

        # ── Dense nodes X_{i,j} for j≥1 ─────────────────────────
        # Input channels = j * filters[i]  (prior nodes)
        #                + filters[i]       (upsampled from below)
        #                = (j+1) * filters[i]
        self.dense = nn.ModuleDict()
        for i in range(depth - 1):              # levels 0, 1, 2
            max_j = depth - 1 - i               # max dense column at this level
            for j in range(1, max_j + 1):
                in_channels = (j + 1) * filters[i]
                self.dense[f"X_{i}_{j}"] = _conv_block(in_channels, filters[i])

        self.final = nn.Conv2d(filters[0], out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        filters = [64, 128, 256, 512]
        depth   = len(filters)

        # ── Encoder pass ─────────────────────────────────────────
        nodes: dict = {}
        xi = x
        for i in range(depth):
            if i > 0:
                xi = self.pool(xi)
            xi = self.enc[i](xi)
            nodes[(i, 0)] = xi

        # ── Dense connections, column by column ──────────────────
        for j in range(1, depth):
            for i in range(depth - j):          # i = 0 .. depth-j-1
                # Collect all prior nodes at this level
                prev     = [nodes[(i, k)] for k in range(j)]
                # Upsample node from the level below (same j-1 column)
                up_feat  = self.up[i](nodes[(i + 1, j - 1)])
                # Guard against 1-pixel rounding from ConvTranspose2d
                if up_feat.shape[-2:] != prev[0].shape[-2:]:
                    up_feat = F.interpolate(
                        up_feat, size=prev[0].shape[-2:],
                        mode="bilinear", align_corners=True,
                    )
                merged         = torch.cat(prev + [up_feat], dim=1)
                nodes[(i, j)]  = self.dense[f"X_{i}_{j}"](merged)

        # ── Output: shallowest level, last dense column ───────────
        return self.final(nodes[(0, depth - 1)])  # (B, 1, H, W)
