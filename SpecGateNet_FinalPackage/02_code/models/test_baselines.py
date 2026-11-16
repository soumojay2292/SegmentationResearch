"""Forward-pass sanity tests for baseline models."""

import torch
from models.attention_unet import AttentionUNet
from models.unet_plus_plus import UNetPlusPlus


def _test(model: torch.nn.Module, name: str, h: int = 384, w: int = 384):
    model.eval()
    x = torch.randn(1, 3, h, w)
    with torch.no_grad():
        y = model(x)
    expected = (1, 1, h, w)
    assert y.shape == expected, f"{name}: expected {expected}, got {tuple(y.shape)}"
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{name:20s}  input {tuple(x.shape)} → output {tuple(y.shape)}  params={params:.1f}M  ✓")


if __name__ == "__main__":
    _test(AttentionUNet(), "AttentionUNet")
    _test(UNetPlusPlus(),  "UNetPlusPlus")
    print("\nAll tests passed.")
