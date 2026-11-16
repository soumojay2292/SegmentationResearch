# -*- coding: utf-8 -*-
# EVALUATION-ONLY per-image metric regeneration for statistical analysis.
# No training, no checkpoint modification. Reuses train.SegDataset + adapter and the
# exact prediction path (sigmoid>0.5, primary head). Region: Dice/IoU via the project's
# compute_metrics formula on single images. Boundary: MedPy hd95/assd (unit spacing,
# 384 grid) + Boundary IoU (Cheng CVPR'21, band 0.02). Writes per-image CSVs and
# verifies aggregates against paper2/results/_boundary_raw.json (which backs Tables I & II).
import os, sys, csv, json, argparse
import numpy as np
import torch

REPO = r"C:/Users/Soumojay/Desktop/New_Sm_P/SegmentationResearch"
os.chdir(REPO); sys.path.insert(0, REPO)

from train import SegDataset, _MAFFNetOriginalAdapter          # exact dataset + recon shim
from medpy.metric.binary import hd95 as medpy_hd95, assd as medpy_assd

SAM2 = "checkpoints/sam2.1_hiera_large.pt"
OUT  = "paper2/results/perimage"
os.makedirs(OUT, exist_ok=True)
DEV  = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Boundary IoU (Cheng et al. 2021), official erosion via cv2 if present else scipy
try:
    import cv2
    def _boundary(mask, ratio=0.02):
        h, w = mask.shape
        d = int(round(ratio * np.sqrt(h*h + w*w))); d = max(d, 1)
        m = cv2.copyMakeBorder(mask, 1,1,1,1, cv2.BORDER_CONSTANT, value=0)
        er = cv2.erode(m, np.ones((3,3), np.uint8), iterations=d)[1:h+1, 1:w+1]
        return mask - er
    BIOU_BACKEND = "cv2"
except Exception:
    from scipy.ndimage import binary_erosion
    _ST = np.ones((3,3), bool)
    def _boundary(mask, ratio=0.02):
        h, w = mask.shape
        d = int(round(ratio * np.sqrt(h*h + w*w))); d = max(d, 1)
        er = binary_erosion(mask.astype(bool), structure=_ST, iterations=d, border_value=0)
        return (mask.astype(bool) & ~er).astype(mask.dtype)
    BIOU_BACKEND = "scipy"

def boundary_iou(gt, dt, ratio=0.02):
    gb = _boundary(gt.astype(np.uint8), ratio) > 0
    db = _boundary(dt.astype(np.uint8), ratio) > 0
    u = (gb | db).sum()
    return float((gb & db).sum() / u) if u > 0 else 1.0   # both empty boundaries -> perfect

DISP2TYPE = {
    "U-Net":"unet", "Attention U-Net":"attention_unet", "UNet++":"unetpp", "TransUNet":"transunet",
    "MAFF-Net (Paper-faithful Recon.)":"recon", "MAFF-Net (Restored)":"maffnet",
    "MAFF-Net (Proposed)":"stabilized",
}

def build(mtype):
    if mtype == "unet":
        from models.unet import UNet; return UNet()
    if mtype == "attention_unet":
        from models.attention_unet import AttentionUNet; return AttentionUNet()
    if mtype == "unetpp":
        from models.unet_plus_plus import UNetPlusPlus; return UNetPlusPlus()
    if mtype == "transunet":
        from models.transunet import TransUNet; return TransUNet()
    if mtype == "maffnet":
        from models.maffnet import MAFFNet; return MAFFNet(checkpoint=SAM2)
    if mtype == "stabilized":
        from models.stabilized_maffnet import StabilizedMAFFNet; return StabilizedMAFFNet(checkpoint=SAM2)
    if mtype == "recon":
        from models.maffnet_original_reconstructed import MAFFNetOriginalReconstructed
        return _MAFFNetOriginalAdapter(MAFFNetOriginalReconstructed(sam_checkpoint=SAM2))
    raise ValueError(mtype)

def load_ckpt(model, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    if all(k.startswith("module.") for k in sd): sd = {k[7:]: v for k, v in sd.items()}
    r = model.load_state_dict(sd, strict=False)
    return len(r.missing_keys), len(r.unexpected_keys)

@torch.no_grad()
def predict_logits(model, x):
    out = model(x)
    return out[0] if isinstance(out, (tuple, list)) else out

def per_image_region(logit, mask):
    prob = torch.sigmoid(logit); pred = (prob > 0.5).float(); tgt = mask
    TP = (pred*tgt).sum().item(); FP = (pred*(1-tgt)).sum().item(); FN = ((1-pred)*tgt).sum().item()
    eps = 1e-8
    dice = 2*TP/(2*TP+FP+FN+eps); iou = TP/(TP+FP+FN+eps)
    return dice, iou, pred, TP, FP, FN

def eval_run(mtype, dataset, ckpt, disp):
    ds = SegDataset(f"dataset_split/{dataset}/test.csv", split="test", img_size=384)
    model = build(mtype).to(DEV).eval()
    mk, uk = load_ckpt(model, ckpt)
    rows = []
    for i in range(len(ds)):
        img, mask = ds[i]
        x = img.unsqueeze(0).to(DEV); m = mask.unsqueeze(0).to(DEV)
        logit = predict_logits(model, x).float()
        dice, iou, pred, TP, FP, FN = per_image_region(logit, m)
        p = pred[0,0].cpu().numpy().astype(np.uint8); g = m[0,0].cpu().numpy().astype(np.uint8)
        empty = int(p.sum() == 0)
        if empty:
            hd = np.nan; asd = np.nan
        else:
            hd = float(medpy_hd95(p, g)); asd = float(medpy_assd(p, g))
        biou = boundary_iou(g, p, 0.02)
        rows.append({"idx": i, "image": ds.samples[i][0], "dice": dice, "iou": iou,
                     "hd95": hd, "assd": asd, "biou": biou, "pred_empty": empty,
                     "tp": TP, "fp": FP, "fn": FN})
    del model; torch.cuda.empty_cache()
    return rows, (mk, uk)

def agg(rows):
    d = np.array([r["dice"] for r in rows]); io = np.array([r["iou"] for r in rows])
    hd = np.array([r["hd95"] for r in rows], float); asd = np.array([r["assd"] for r in rows], float)
    bi = np.array([r["biou"] for r in rows]); ne = sum(r["pred_empty"] for r in rows)
    v = ~np.isnan(hd)
    # EXACT paper aggregation: compute_metrics on bs=2 batches, batch-weighted mean
    # (loader order == sequential idx, no shuffle). All test n even -> clean pairs.
    eps = 1e-8; pdice = []; piou = []
    for j in range(0, len(rows), 2):
        grp = rows[j:j+2]
        TP = sum(r["tp"] for r in grp); FP = sum(r["fp"] for r in grp); FN = sum(r["fn"] for r in grp)
        pdice.append(2*TP/(2*TP+FP+FN+eps)); piou.append(TP/(TP+FP+FN+eps))
    return dict(n=len(rows), dice=d.mean(), iou=io.mean(),
               paper_dice=float(np.mean(pdice)), paper_iou=float(np.mean(piou)),
               hd95=np.nanmean(hd), assd=np.nanmean(asd), biou=bi.mean(),
               n_valid=int(v.sum()), n_empty=ne)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    runs = json.load(open("paper2/results/_boundary_raw.json"))["runs"]
    ref = {(r["model"], r["dataset"]): r for r in runs}
    canon = [(r["model"], r["dataset"], r["checkpoint"]) for r in runs]
    if a.smoke:
        canon = [c for c in canon if c[0]=="MAFF-Net (Proposed)" and c[1]=="ISIC_2018"]
    print(f"BIoU backend: {BIOU_BACKEND} | device: {DEV} | runs: {len(canon)}\n")
    print(f"{'model':34s}{'dset':10s}{'dice(d)':16s}{'hd95(d)':16s}{'assd(d)':14s}{'biou(d)':14s}{'nv/ne':9s}{'keys(m/u)'}")
    allok = True
    for disp, dset, ckpt in canon:
        rows, (mk, uk) = eval_run(DISP2TYPE[disp], dset, ckpt, disp)
        A = agg(rows); R = ref[(disp, dset)]
        dd = A["paper_dice"]-R["dice_recomputed"]; dh = A["hd95"]-R["hd95_mean"]
        da = A["assd"]-R["assd_mean"]; db = A["biou"]-R["biou_mean"]
        okv = (A["n_valid"]==R["n_valid_distance"] and A["n_empty"]==R["n_pred_empty"])
        ok = abs(dd)<2e-3 and abs(dh)<0.05 and abs(da)<0.05 and abs(db)<3e-3 and okv
        allok &= ok
        # write per-image csv
        safe = disp.replace(" ","_").replace("(","").replace(")","").replace(".","").replace("/","_")
        with open(f"{OUT}/{safe}__{dset}.csv","w",newline="") as fh:
            w=csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        flag = "" if ok else "  <-- CHECK"
        print(f"{disp:34s}{dset:10s}{A['paper_dice']:.4f}({dd:+.4f})  {A['hd95']:7.3f}({dh:+.3f})  "
              f"{A['assd']:6.3f}({da:+.3f})  {A['biou']:.4f}({db:+.4f})  {A['n_valid']}/{A['n_empty']}  {mk}/{uk}{flag}")
    print("\nALL WITHIN TOLERANCE:" , allok)

if __name__ == "__main__":
    main()
