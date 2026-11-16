"""
Ablation variants of the Proposed MAFF-Net (models/specgate_net.py).

SEPARATE, NON-INVASIVE subclass used ONLY for the component ablation study. It
NEVER modifies SpecGateNet. Each instance disables exactly ONE architectural
contribution by swapping the corresponding submodule(s) for a shape-preserving,
parameter-free stand-in, leaving the frozen SAM2 encoder, the decoder, the RDB, the
segmentation heads, and the training/loss pipeline untouched.

Leave-one-out ablations (from the Full Proposed model):
    "freq"   - Frequency Branch  -> nn.Identity()
               The frequency stream becomes align_f[i](Identity(f)) = align_f[i](f):
               a second aligned view of the raw feature, with no spectral high-pass.
    "align"  - Feature Alignment -> nn.Identity() on BOTH streams
               fs = f and ff = freq[i](f) reach the gate unnormalised. Shape-safe
               because FeatureAlignment maps c -> c.
    "fusion" - Gated Fusion      -> MeanFusion (parameter-free 0.5*(Fs + Ff))
               The learned per-pixel gate is replaced by a fixed unweighted mean.

Output interface is unchanged: forward returns (p1, p2, p3, p4, rec), so the
canonical MAFFNetTrainer / MAFFNetLoss run without modification.
"""

import torch.nn as nn

from models.specgate_net import SpecGateNet


class MeanFusion(nn.Module):
    """Parameter-free replacement for GatedFusion: out = 0.5*(F_spatial + F_freq).

    Matches GatedFusion's forward signature (two equal-width tensors -> one), so it
    is a drop-in inside the parent forward without any forward override.
    """

    def forward(self, f_spatial, f_freq):
        return 0.5 * (f_spatial + f_freq)


class SpecGateNetAblation(SpecGateNet):
    """SpecGateNet with exactly one architectural component disabled.

    Args:
        encoder / checkpoint / cutoff: passed straight to SpecGateNet.
        ablate: which contribution to disable — one of {"freq", "align", "fusion"}.

    Everything else — the frozen backbone, skip projections, decoder, heads, RDB —
    is inherited unchanged from the canonical model.
    """

    VALID = ("freq", "align", "fusion")

    def __init__(self, encoder=None, checkpoint: str = None,
                 cutoff: float = 0.1, ablate: str = None):
        if ablate not in self.VALID:
            raise ValueError(f"ablate must be one of {self.VALID}, got {ablate!r}")

        super().__init__(encoder=encoder, checkpoint=checkpoint, cutoff=cutoff)

        n = len(self.freq)          # number of encoder scales (4)
        self.ablate = ablate

        if ablate == "freq":
            self.freq = nn.ModuleList([nn.Identity() for _ in range(n)])
        elif ablate == "align":
            self.align_s = nn.ModuleList([nn.Identity() for _ in range(n)])
            self.align_f = nn.ModuleList([nn.Identity() for _ in range(n)])
        elif ablate == "fusion":
            self.fuse = nn.ModuleList([MeanFusion() for _ in range(n)])
