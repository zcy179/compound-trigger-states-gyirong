"""Pair the calibrated perfect-prog track against the archived GFS and GEFS tracks
on the held-out issues at D = 3, using a single archive chain throughout.

Why this script exists
----------------------
Earlier drafts of Section 4.5 mixed three archives: the PP side came from
`work/results/p3_hindcast_cases.csv` (the 05_p3.py three-day track, whose AR(1)
error model was fitted without the 19 August 2026 cut-off), the GEFS side came from
`work/results/gefs_hindcast_cases_z3904.csv` (the 3904 m elevation variant), and the
reported differences did not equal the difference of the reported levels.
Table 6, Table 8b and Supplementary Table S10 all use the main H = 7 track
(`work/analysis/P1_A6_h7_hindcast_cases.csv`, CUTOFF = 2026-08-19). This script
reproduces every number of Section 4.5 from that main track instead.

Outputs
-------
work/results/p1_a_paired_main_track.csv   point estimates, intervals, counts
work/analysis/P1_A8_costloss_main_track.csv  cost-loss relative value on the same pairs

Conventions
-----------
* BSS is relative to the climatology of the scored sample, as in Tables 6, 7, 11.
* POD / FAR use the fixed screening threshold p >= 0.05 of Section 3.5, with
  FAR = FP / (TP + FP) (the WMO false-alarm ratio, the definition used by
  work/analysis/P2_holdout_reanalysis.py).
* BSS intervals: 10-day block bootstrap, 1,000 resamples, seed 20261030.
* AUC interval and p-value: DeLong for paired ROC curves (DeLong et al., 1988).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H7 = os.path.join(ROOT, "work", "analysis", "P1_A6_h7_hindcast_cases.csv")
GFS = os.path.join(ROOT, "work", "results", "gfs_hindcast_cases.csv")
GEFS = os.path.join(ROOT, "work", "results", "gefs_hindcast_cases.csv")
OUT = os.path.join(ROOT, "work", "results", "p1_a_paired_main_track.csv")
COSTLOSS = os.path.join(ROOT, "work", "analysis", "P1_A8_costloss_main_track.csv")
GFS_COSTLOSS = os.path.join(ROOT, "work", "analysis", "P1_A8_costloss.csv")

D = 3
THR = 0.05
SEED = 20261030
NBOOT = 1000
BLOCK = 10


def bss(p, y):
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    return 1.0 - np.mean((p - y) ** 2) / np.mean((y.mean() - y) ** 2)


def podfar(p, y, thr=THR):
    warn = np.asarray(p, float) >= thr
    y = np.asarray(y) == 1
    tp = int((warn & y).sum())
    fp = int((warn & ~y).sum())
    fn = int((~warn & y).sum())
    pod = tp / (tp + fn) if (tp + fn) else np.nan
    far = fp / (tp + fp) if (tp + fp) else np.nan
    return pod, far


def midrank(x):
    order = np.argsort(x)
    z = np.asarray(x, float)[order]
    n = len(z)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def delong_paired(score_a, score_b, y):
    """DeLong test for two correlated ROC areas on the same cases."""
    y = np.asarray(y).astype(int)
    pos = y == 1
    neg = ~pos
    m, n = int(pos.sum()), int(neg.sum())

    def components(s):
        s = np.asarray(s, float)
        tx, ty, tz = midrank(s[pos]), midrank(s[neg]), midrank(s)
        v10 = (tz[pos] - tx) / n
        v01 = 1.0 - (tz[neg] - ty) / m
        return v10, v01

    a10, a01 = components(score_a)
    b10, b01 = components(score_b)
    auc_a, auc_b = a10.mean(), b10.mean()
    s10 = np.cov(np.vstack([a10, b10])) / m
    s01 = np.cov(np.vstack([a01, b01])) / n
    cov = s10 + s01
    se = float(np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]))
    diff = float(auc_a - auc_b)
    z = diff / se if se > 0 else np.nan
    from math import erfc, sqrt

    p = erfc(abs(z) / sqrt(2)) if np.isfinite(z) else np.nan
    return diff, se, 1.96 * se, p


def block_bootstrap(n, stats, seed=SEED, nboot=NBOOT, block=BLOCK):
    """Resample contiguous blocks of consecutive issue dates with replacement."""
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = np.arange(n)
    draws = []
    for _ in range(nboot):
        pick = rng.choice(starts, size=nb, replace=True)
        idx = np.concatenate([(np.arange(s, s + block) % n) for s in pick])[:n]
        vals = [f(idx) for f in stats]
        if all(np.isfinite(v) for v in vals):
            draws.append(vals)
    arr = np.array(draws)
    return np.percentile(arr, 2.5, axis=0), np.percentile(arr, 97.5, axis=0), len(arr)


def costloss(p, y, grid):
    """Richardson (2000) relative value, warning when p >= C/L."""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    base = float(y.mean())
    rows = []
    for cl in grid:
        warn = p >= cl
        nw = int(warn.sum())
        tp = int((warn & (y == 1)).sum())
        fp = int((warn & (y == 0)).sum())
        pod = tp / y.sum()
        far = fp / (tp + fp) if tp + fp else np.nan
        csi = tp / (tp + fp + int((~warn & (y == 1)).sum()))
        e_fc = cl * nw / len(y) + base * (1 - pod)
        e_clim = min(cl, base)
        e_perf = base * cl
        rv = (e_clim - e_fc) / (e_clim - e_perf) if e_clim > e_perf else np.nan
        rows.append(dict(CL=cl, n=len(y), base=base, nwarn=nw, pod=pod, far=far,
                         csi=csi, RV=rv))
    return pd.DataFrame(rows)


def main():
    h7 = pd.read_csv(H7, parse_dates=["issue"])
    gfs = pd.read_csv(GFS, parse_dates=["issue"])
    gefs = pd.read_csv(GEFS, parse_dates=["issue"])

    h7 = h7[h7.year >= 2024].copy()
    pp_col, pp_raw = f"pc_d{D}", f"p_d{D}"
    y_col = f"y_d{D}"

    g = gfs[["issue", "year", f"p_d{D}", f"y_d{D}"]].rename(
        columns={f"p_d{D}": "gfs_p", f"y_d{D}": "gfs_y"}
    )
    m = h7[["issue", pp_col, pp_raw, y_col]].merge(g, on="issue").sort_values("issue")
    m = m.reset_index(drop=True)

    y = m[y_col].to_numpy().astype(float)
    assert np.array_equal(y, m["gfs_y"].to_numpy().astype(float)), "label mismatch"
    assert len(m) == 437, len(m)

    pp, pp_r, gg = m[pp_col].to_numpy(float), m[pp_raw].to_numpy(float), m["gfs_p"].to_numpy(float)

    pp_bss, gfs_bss = bss(pp, y), bss(gg, y)
    pp_auc, gfs_auc = roc_auc_score(y, pp), roc_auc_score(y, gg)
    pp_pod, pp_far = podfar(pp, y)
    gfs_pod, gfs_far = podfar(gg, y)

    # The GFS side was published before this script existed; the point scores must
    # come out identical to four decimals or the pairing assumption is wrong.
    assert round(gfs_bss, 4) == 0.3327, gfs_bss
    assert round(gfs_auc, 4) == 0.9502, gfs_auc
    assert round(pp_bss, 4) == 0.7667, pp_bss
    assert round(pp_auc, 3) == 0.996, pp_auc

    def stat_bss(idx):
        return bss(pp[idx], y[idx]) - bss(gg[idx], y[idx])

    lo, hi, kept = block_bootstrap(len(m), [stat_bss])

    auc_diff, auc_se, auc_ci, auc_p = delong_paired(pp, gg, y)

    # GEFS: same engine, inflation-corrected ensemble, 147 overlapping issues.
    e = gefs[["issue", "year", f"p_d{D}", f"pinf_d{D}", f"y_d{D}"]].rename(
        columns={f"p_d{D}": "gefs_p", f"pinf_d{D}": "gefs_pinf", f"y_d{D}": "gefs_y"}
    )
    mg = h7[["issue", pp_col, y_col]].merge(e, on="issue").sort_values("issue")
    mg = mg[mg.year >= 2024].reset_index(drop=True)
    yg = mg[y_col].to_numpy(float)
    assert np.array_equal(yg, mg["gefs_y"].to_numpy().astype(float)), "GEFS label mismatch"

    rows = [
        dict(comparison="perfect-prog vs GFS", horizon="D=%d" % D, n=len(m),
             positives=int(y.sum()), base_rate=round(float(y.mean()), 4),
             bss_first=round(pp_bss, 4), bss_second=round(gfs_bss, 4),
             bss_diff=round(pp_bss - gfs_bss, 4),
             bss_lo=round(float(lo[0]), 4), bss_hi=round(float(hi[0]), 4),
             auc_first=round(pp_auc, 4), auc_second=round(gfs_auc, 4),
             auc_diff=round(auc_diff, 4), auc_lo=round(auc_diff - auc_ci, 4),
             auc_hi=round(auc_diff + auc_ci, 4), auc_p=round(float(auc_p), 6),
             pod_first=round(pp_pod, 3), pod_second=round(gfs_pod, 3),
             far_first=round(pp_far, 3), far_second=round(gfs_far, 3),
             note="bootstrap draws kept %d/%d; seed %d; 10-day blocks" % (kept, NBOOT, SEED)),
        dict(comparison="perfect-prog vs GEFS (inflation-corrected)", horizon="D=%d" % D,
             n=len(mg), positives=int(yg.sum()), base_rate=round(float(yg.mean()), 4),
             bss_first=round(bss(mg[pp_col], yg), 4),
             bss_second=round(bss(mg["gefs_pinf"], yg), 4),
             bss_diff=round(bss(mg[pp_col], yg) - bss(mg["gefs_pinf"], yg), 4),
             auc_first=round(roc_auc_score(yg, mg[pp_col]), 4),
             auc_second=round(roc_auc_score(yg, mg["gefs_pinf"]), 4),
             pod_first=np.nan, pod_second=np.nan, far_first=np.nan, far_second=np.nan,
             note="GEFS available on %d of the 437 issues" % len(mg)),
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    # Cost-loss on the same 437 pairs, both tracks recomputed here so the two curves
    # come from one code path (the archived P1_A8_costloss.csv used the old PP track).
    grid = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30,
            0.40, 0.50, 0.60, 0.70, 0.80]
    cl = []
    for tag, prob in (("perfect-prog (calibrated)", pp), ("GFS (raw, 3-day)", gg)):
        d = costloss(prob, y, grid)
        d.insert(0, "model", tag)
        cl.append(d)
    cl = pd.concat(cl, ignore_index=True)
    cl.to_csv(COSTLOSS, index=False)

    # The GFS curve must reproduce the archived one row for row.
    old = pd.read_csv(GFS_COSTLOSS)
    old = old[old.model.str.startswith("GFS")].reset_index(drop=True)
    new = cl[cl.model.str.startswith("GFS")].reset_index(drop=True)
    gap = float(np.max(np.abs(old.RV.to_numpy() - new.RV.to_numpy())))
    print("GFS cost-loss rows reproduced, max |RV difference| = %.2e" % gap)
    assert gap < 5e-4, gap

    print(out.T.to_string())
    print()
    print("PP raw (uncalibrated) at D=%d: BSS %.4f  AUC %.4f" % (D, bss(pp_r, y), roc_auc_score(y, pp_r)))
    print("PP cost-loss RV: %.3f (C/L=0.005)  %.3f (C/L=0.30)" % (
        float(cl[(cl.model.str.startswith("perfect")) & (cl.CL == 0.005)].RV.iloc[0]),
        float(cl[(cl.model.str.startswith("perfect")) & (cl.CL == 0.30)].RV.iloc[0])))
    print("wrote", OUT)
    print("wrote", COSTLOSS)


if __name__ == "__main__":
    main()
