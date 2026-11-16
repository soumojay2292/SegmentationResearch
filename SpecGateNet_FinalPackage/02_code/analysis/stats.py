# -*- coding: utf-8 -*-
# Paired statistical analysis on regenerated per-image metrics.
# Friedman omnibus (7 models) + post-hoc Wilcoxon signed-rank (Proposed vs each other
# model) with manual Holm correction, matched-pairs rank-biserial effect size, and
# bootstrap 95% CI of the median paired difference. Distance metrics use complete-case
# pairing (images where both models produced a non-empty prediction).
import os, csv, json
import numpy as np
from scipy import stats

REPO = r"C:/Users/Soumojay/Desktop/New_Sm_P/SegmentationResearch"
os.chdir(REPO)
PER = "paper2/results/perimage"; OUT = "paper2/results/stats"; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20260826)

MODELS = ["U-Net","Attention U-Net","UNet++","TransUNet",
          "MAFF-Net (Paper-faithful Recon.)","MAFF-Net (Restored)","MAFF-Net (Proposed)"]
PROP = "MAFF-Net (Proposed)"
DSETS = ["ISIC_2016","ISIC_2017","ISIC_2018"]
# metric: (column, higher_is_better)
METRICS = [("dice",True),("iou",True),("hd95",False),("assd",False),("biou",True)]

def safe(d): return d.replace(" ","_").replace("(","").replace(")","").replace(".","").replace("/","_")

def load(dset):
    """Return {model: {col: np.array indexed by image idx}} aligned by idx."""
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
        # complete-case mask across all 7 (for Friedman); for pairwise, per-pair below
        M = np.vstack([mats[m] for m in MODELS])           # 7 x n
        keep_all = ~np.isnan(M).any(axis=0)
        Mc = M[:, keep_all]
        chi, p = stats.friedmanchisquare(*[Mc[i] for i in range(len(MODELS))])
        fried.append(dict(dataset=dset, metric=col, higher_better=hib,
                          n_complete=int(keep_all.sum()), friedman_chi2=chi, friedman_p=p))
        # pairwise Proposed vs each other
        praw = []; recs = []
        xP = mats[PROP]
        for m in MODELS:
            if m == PROP: continue
            xO = mats[m]
            pair = ~np.isnan(xP) & ~np.isnan(xO)
            a = xP[pair]; b = xO[pair]; diff = a - b
            nz = np.count_nonzero(diff)
            if nz == 0:
                W = np.nan; pv = 1.0
            else:
                W, pv = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
            r = rank_biserial(diff)
            lo, hi = boot_ci_median(diff)
            recs.append(dict(dataset=dset, metric=col, higher_better=hib, comparison=f"Proposed vs {m}",
                             n_pairs=int(pair.sum()), median_diff=float(np.median(diff)) if pair.sum() else np.nan,
                             ci95_lo=lo, ci95_hi=hi, wilcoxon_W=float(W) if not np.isnan(W) else np.nan,
                             p_raw=float(pv), rank_biserial=r))
            praw.append(pv)
        padj = holm(np.array(praw))
        for rec, pa in zip(recs, padj):
            rec["p_holm"] = float(pa); rec["sig_holm_.05"] = bool(pa < 0.05)
            post.append(rec)

# write CSVs
def wcsv(path, rows):
    with open(path,"w",newline="") as fh:
        w=csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
wcsv(f"{OUT}/friedman.csv", fried)
wcsv(f"{OUT}/posthoc_proposed.csv", post)

# console summary
print("=== FRIEDMAN (7 models) ===")
for r in fried:
    print(f"  {r['dataset']} {r['metric']:5s} n={r['n_complete']:3d}  chi2={r['friedman_chi2']:8.2f}  p={r['friedman_p']:.2e}")
print("\n=== POST-HOC Wilcoxon: Proposed vs others (Holm) — key rows ===")
for dset in DSETS:
    for col,_ in METRICS:
        sub=[r for r in post if r["dataset"]==dset and r["metric"]==col]
        for m in ["MAFF-Net (Restored)","MAFF-Net (Paper-faithful Recon.)","TransUNet"]:
            r=[x for x in sub if x["comparison"]==f"Proposed vs {m}"][0]
            sig = "sig" if r["sig_holm_.05"] else "ns "
            print(f"  {dset} {col:5s} vs {m:32s} n={r['n_pairs']:3d} md={r['median_diff']:+.4f} "
                  f"CI[{r['ci95_lo']:+.4f},{r['ci95_hi']:+.4f}] p_holm={r['p_holm']:.2e} {sig} r={r['rank_biserial']:+.2f}")
print(f"\nWrote {OUT}/friedman.csv and {OUT}/posthoc_proposed.csv")
