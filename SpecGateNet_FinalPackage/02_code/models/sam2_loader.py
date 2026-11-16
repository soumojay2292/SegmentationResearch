"""
SAM2 loader — bypasses Hydra initialize() entirely.
Uses OmegaConf.load + hydra.utils.instantiate directly.
Drop-in replacement for build_sam2().
"""
import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

def setup_sam2_path(sam2_repo: str = "src/sam2"):
    p = str(Path(sam2_repo).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def load_sam2_encoder(
    config_path: str = "src/sam2/sam2/configs/sam2_hiera_l.yaml",
    checkpoint:  str = "sam2_hiera_large.pt",
    device:      str = "cpu",
):
    """
    Returns sam2.image_encoder (Hiera-large), frozen.
    No Hydra initialize() — loads config via OmegaConf directly.
    """

    cfg_path = Path(config_path).resolve()
    if not cfg_path.exists():
        # try alternate location (sam2.1 repos renamed configs dir)
        alt = Path(config_path.replace("/sam2/", "/sam2.1/")).resolve()
        if alt.exists():
            cfg_path = alt
        else:
            raise FileNotFoundError(
                f"SAM2 config not found.\n  tried: {cfg_path}\n  tried: {alt}\n"
                f"  run: find src/sam2 -name '*.yaml' | grep hiera"
            )

    cfg = OmegaConf.load(cfg_path)

    # instantiate full SAM2Base model from _target_ keys in yaml
    model = instantiate(cfg.model, _recursive_=True)
    model = model.to(device)
    model.eval()

    ckpt_path = Path(checkpoint)
    if ckpt_path.exists():
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[SAM2] {len(missing)} missing keys (expected for encoder-only use)")
        if unexpected:
            print(f"[SAM2] {len(unexpected)} unexpected keys")
    else:
        print(f"[SAM2] WARNING: checkpoint not found at {checkpoint} — random weights")

    return model.image_encoder