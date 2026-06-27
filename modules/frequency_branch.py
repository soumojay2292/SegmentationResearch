"""
Frequency Branch module (synopsis §1.4 — stabilised MAFFNet).

Extracts a boundary-sensitive spatial feature map by retaining only the
high-frequency content of the input:

    x  ->  rfft2 (fp32)  ->  radial high-pass mask  ->  irfft2 (fp32)  ->  out

The FFT is computed in single precision for numerical stability (half-precision
FFT overflows under AMP). The high-pass mask is a fixed (non-learnable) binary
radial filter with a configurable cutoff. Output preserves the input spatial
resolution and is cast back to the input dtype.

Standalone and reusable: depends only on torch. Not integrated anywhere yet.
Does not import or modify the current MAFFNet implementation or the trainer.
Note: this branch uses no abs()/sqrt() on the spectrum, so its backward is
finite everywhere (unlike a magnitude-based path).
"""

import torch
import torch.nn as nn


class FrequencyBranch(nn.Module):
    """fp32 FFT -> radial high-pass -> inverse FFT.

    Args:
        cutoff: radial cutoff in normalized frequency (cycles/pixel). Frequencies
                with radius r >= cutoff are kept, r < cutoff are removed.
                Valid range ~[0, 0.707]; default 0.1. Larger cutoff = more
                aggressive high-pass (keeps only sharper edges).

    Shape:
        input:  (B, C, H, W)
        output: (B, C, H, W)   (same resolution, real-valued)
    """

    def __init__(self, cutoff: float = 0.1):
        super().__init__()
        self.cutoff = float(cutoff)
        # Fixed (non-learnable) mask, cached per spatial size + device.
        self._mask_cache = {}

    def _highpass_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (H, W, str(device))
        mask = self._mask_cache.get(key)
        if mask is None:
            fy = torch.fft.fftfreq(H, device=device)      # (H,)        in [-0.5, 0.5)
            fx = torch.fft.rfftfreq(W, device=device)     # (W//2+1,)   in [0, 0.5]
            r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)   # (H, W//2+1)
            mask = (r >= self.cutoff).to(torch.float32)   # binary radial high-pass
            self._mask_cache[key] = mask
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2], x.shape[-1]
        # fp32 forward FFT (stable under mixed precision)
        xf = torch.fft.rfft2(x.float(), norm="ortho")     # complex (B, C, H, W//2+1)
        mask = self._highpass_mask(H, W, x.device)        # (H, W//2+1), broadcast over B,C
        xf = xf * mask
        # inverse FFT back to the original spatial resolution
        out = torch.fft.irfft2(xf, s=(H, W), norm="ortho")  # real (B, C, H, W)
        return out.to(x.dtype)
