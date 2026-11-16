# -*- coding: utf-8 -*-
# 6-model recompute (Paper-faithful Reconstruction removed). Same methodology as stats.py.
import os, csv
import numpy as np
from scipy import stats

REPO = r"C:/Users/Soumojay/Desktop/New_Sm_P/SegmentationResearch"
os.chdir(REPO)
PER = "paper2/results/perimage"; OUT = "paper2/results/stats6"; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20260826)

MODELS = ["U-Net","Attention U-Net","UNet++","TransUNet",
          "MAFF-Net (Restored)","MAFF-Net (Proposed)"]   # Paper-faithful dropped
PROP = "MAFF-Net (Proposed)"; REST = "MAFF-Net (Restored)"
BASELINES = ["U-Net","Attention U-Net","UNet++","TransUNet"]
DSETS = ["ISIC_2016","ISIC_2017","ISIC_2018"]
METRICS = [("dice",True),("iou",True),("hd95",False),("assd",False),("biou",True)]

def safe(d): return d.replace(" ","_").replace("(","").replace(")","").replace(".","").replace("/","_")

def load(dset):
    data = {}
    for m in MODELS:
        rows = list(csv.DictReader(open(f"{PER}/{safe(m)}__{dset}.csv")))
        rows.sort(key=lambda r: int(r["idx"]))
        d = {"idx":[int(r["idx"]) for r in rows]}
        for col in ["dice","iou","hd95","assd","biou"]:
            d[col] = np.array([float(r[col]) if r[col] not in ("","nan") else np.nan for r in rows])
        data[m] = d
    idx0 = data[MODELS[0]]["idx"]
    for m in MODELS: assert data[m]["idx"] == idx0, f"idx mismatch {m} {dset}"
    return data, len(idx0)

def holm(pvals):
    m = len(pvals); order = np.argsort(pvals); adj = np.empty(m); prev = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        prev = max(prev, min(val, 1.0)); adj[i] = prev
    return adj

def rank_biserial(diff):
    d = diff[diff != 0]
    if len(d) == 0: return 0.0
    r = stats.rankdata(np.abs(d)); Rp = r[d>0].sum(); Rn = r[d<0].sum()
    return float((Rp - Rn)/(Rp + Rn))

def boot_ci_median(diff, B=10000):
    if len(diff) == 0: return (np.nan, np.nan)
    idx = rng.integers(0, len(diff), size=(B, len(diff)))
    meds = np.median(diff[idx], axis=1)
    return float(np.percentile(meds,2.5)), float(np.percentile(meds,97.5))

fried = []; post = []
for dset in DSETS:
    data, n = load(dset)
    for col, hib in METRICS:
        mats = {m: data[m][col] for m in MODELS}
        M = np.vstack([mats[m] for m in MODELS])
        keep_all = ~np.isnan(M).any(axis=0)
        Mc = M[:, keep_all]
        chi, p = stats.friedmanchisquare(*[Mc[i] for i in range(len(MODELS))])
        fried.append(dict(dataset=dset, metric=col, higher_better=hib,
                          n_complete=int(keep_all.sum()), friedman_chi2=chi, friedman_p=p))
        praw = []; recs = []
        xP = mats[PROP]
        for m in MODELS:
            if m == PROP: continue
            xO = mats[m]
            pair = ~np.isnan(xP) & ~np.isnan(xO)
            a = xP[pair]; b = xO[pair]; diff = a - b
            nz = np.count_nonzero(diff)
            if nz == 0: W = np.nan; pv = 1.0
            else: W, pv = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
            r = rank_biserial(diff); lo, hi = boot_ci_median(diff)
            recs.append(dict(dataset=dset, metric=col, higher_better=hib, comparison=f"Proposed vs {m}",
                             n_pairs=int(pair.sum()), median_diff=float(np.median(diff)) if pair.sum() else np.nan,
                             ci95_lo=lo, ci95_hi=hi, wilcoxon_W=float(W) if not np.isnan(W) else np.nan,
                             p_raw=float(pv), rank_biserial=r))
            praw.append(pv)
        padj = holm(np.array(praw))
        for rec, pa in zip(recs, padj):
            rec["p_holm"] = float(pa); rec["sig_holm_.05"] = bool(pa < 0.05)
            post.append(rec)

def wcsv(path, rows):
    with open(path,"w",newline="") as fh:
        w=csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
wcsv(f"{OUT}/friedman.csv", fried); wcsv(f"{OUT}/posthoc_proposed.csv", post)

# ---- report exactly what the manuscript needs ----
maxfp = max(r["friedman_p"] for r in fried)
print("FRIEDMAN 6-model: max p across all dataset*metric = %.3e ; all < 1e-39: %s" % (maxfp, maxfp < 1e-39))

# baseline bound: max Holm p and min |r| across SpecGate-vs-baseline over all dset*metric
bl = [r for r in post if r["comparison"].split("vs ")[1] in BASELINES]
print("BASELINES: max p_Holm = %.3e ; min |r| = %.3f ; all sig@.05: %s"
      % (max(r["p_holm"] for r in bl), min(abs(r["rank_biserial"]) for r in bl), all(r["sig_holm_.05"] for r in bl)))

# Table IV: SpecGate vs MAFF-Net [15] (Restored): median Δ, r, p_Holm, sig
print("\nTABLE IV  (vs MAFF-Net [15]): dataset metric  medianD    r     p_Holm    sig")
DMAP={"ISIC_2016":"2016","ISIC_2017":"2017","ISIC_2018":"2018"}
for dset in DSETS:
    for col,_ in METRICS:
        r=[x for x in post if x["dataset"]==dset and x["metric"]==col and x["comparison"]==f"Proposed vs {REST}"][0]
        print("  %s %-5s  %+.4f  %+.2f  %.3e  %s" % (DMAP[dset], col, r["median_diff"], r["rank_biserial"], r["p_holm"], "*" if r["sig_holm_.05"] else ""))
