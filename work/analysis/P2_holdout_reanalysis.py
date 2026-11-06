# -*- coding: utf-8 -*-
"""
P2: 基于既有存档的留出重算与基线补算（不重跑集合、不重拟合 isotonic、不生成新预报）。

输入（既有存档，只读）：
  work/analysis/P1_A6_h7_hindcast_cases.csv   issue, year, p_d1..p_d7, y_d1..y_d7, pc_d1..pc_d7
  work/results/p3_hindcast_cases.csv          issue, cti0（起报日观测 CTI）

说明：
  - isotonic 映射与 AR(1) 误差模型参数均拟合于 2021-2023，未做任何改动。
  - 本脚本只做：① 按起报日切分 cal/test 子集并重算评分；② 持续性基线；
    ③ episode 覆盖的二项置信区间。所有数字均可由上述两个存档复算。

输出：
  work/analysis/P2_holdout_pp_metrics.csv
  work/analysis/P2_persistence_baseline.csv
  work/analysis/P2_episode_binomial_ci.csv
"""
import numpy as np
import pandas as pd

RNG_SEED = 20261030
N_BOOT = 1000
BLOCK = 10
C_STAR = 2.5

h = pd.read_csv("P1_A6_h7_hindcast_cases.csv", parse_dates=["issue"]).sort_values("issue").reset_index(drop=True)
c = pd.read_csv("../results/p3_hindcast_cases.csv", parse_dates=["issue"])[["issue", "cti0"]]
h = h.merge(c, on="issue", how="left")


# ----------------------------------------------------------------- scores
def bss(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    bs = np.mean((p - y) ** 2)
    ybar = y.mean()
    return bs, 1 - bs / (ybar * (1 - ybar))


def auc(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().values
    n1, n0 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def podfar(p, y, thr=0.05):
    p, y = np.asarray(p, float), np.asarray(y, float)
    w = p >= thr
    hits = int((w & (y == 1)).sum())
    fal = int((w & (y == 0)).sum())
    miss = int((~w & (y == 1)).sum())
    pod = hits / (hits + miss) if (hits + miss) else np.nan
    far = fal / (hits + fal) if (hits + fal) else np.nan
    return pod, far


def blocks_of(sub, block=BLOCK):
    out, cur_year, cur = [], None, []
    for pos, yr in enumerate(sub["year"].values):
        if yr != cur_year:
            if cur:
                out.append(cur)
            cur_year, cur = yr, []
        cur.append(pos)
        if len(cur) == block:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def block_bootstrap(p, y, blk, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    p, y = np.asarray(p, float), np.asarray(y, float)
    nb = len(blk)
    bs_l, au_l = [], []
    for _ in range(n_boot):
        ii = np.concatenate([blk[k] for k in rng.integers(0, nb, nb)])
        pp, yy = p[ii], y[ii]
        if yy.sum() == 0:
            continue
        _, b = bss(pp, yy)
        bs_l.append(b)
        au_l.append(auc(pp, yy))
    q = lambda a, v: float(np.percentile(a, v)) if len(a) else np.nan
    return q(bs_l, 2.5), q(bs_l, 97.5), q(au_l, 2.5), q(au_l, 97.5)


SUBSETS = {
    "pooled_2021_2026": h["year"].notna(),
    "cal_2021_2023": h["year"] <= 2023,
    "test_2024_2026": h["year"] >= 2024,
}

rows = []
for sname, mask in SUBSETS.items():
    sub = h[mask].reset_index(drop=True)
    blk = blocks_of(sub)
    for track, pref in [("raw", "p_d"), ("isotonic", "pc_d")]:
        for D in range(1, 8):
            p = sub[f"{pref}{D}"].values
            y = sub[f"y_d{D}"].values
            bs, b = bss(p, y)
            pod, far = podfar(p, y)
            blo, bhi, alo, ahi = block_bootstrap(p, y, blk)
            rows.append(dict(subset=sname, track=track, D=D, n=len(sub),
                             positives=int(y.sum()), base_rate=float(y.mean()),
                             brier=float(bs), bss=float(b), bss_lo=blo, bss_hi=bhi,
                             auc=float(auc(p, y)), auc_lo=alo, auc_hi=ahi,
                             pod=pod, far=far, n_blocks=len(blk)))
pp = pd.DataFrame(rows)
pp.to_csv("P2_holdout_pp_metrics.csv", index=False, encoding="utf-8-sig")


# --------------------------------------------------- persistence baseline
# 基线定义：若起报日观测 CTI >= C*，则预报该 D 天窗口内出现红态（p = 1），否则 p = 0。
# 注意：cti0 用到起报日全天最大 VPD，而预报引擎不使用该量，故该基线偏强（保守）。
rows = []
for sname, mask in SUBSETS.items():
    sub = h[mask].reset_index(drop=True)
    p_pers = (sub["cti0"].values >= C_STAR).astype(float)
    for D in range(1, 8):
        y = sub[f"y_d{D}"].values
        bs, b = bss(p_pers, y)
        pod, far = podfar(p_pers, y)
        rows.append(dict(subset=sname, D=D, n=len(sub), positives=int(y.sum()),
                         base_rate=float(y.mean()), n_warn=int((p_pers >= 0.5).sum()),
                         brier=float(bs), bss=float(b), auc=float(auc(p_pers, y)),
                         pod=pod, far=far))
pb = pd.DataFrame(rows)
pb.to_csv("P2_persistence_baseline.csv", index=False, encoding="utf-8-sig")


# ----------------------------------------------- episode coverage binomial CI
def wilson(k, n, z=1.959963985):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    den = 1 + z * z / n
    cen = (ph + z * z / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, cen - half), min(1.0, cen + half))


rows = []
for tag, k, n in [("episode coverage D=7 (7-day config)", 3, 7),
                  ("episode coverage D=7 (3-day config)", 6, 7),
                  ("episode coverage D=1 (7-day config)", 7, 7),
                  ("episode coverage D=4 (7-day config)", 6, 7),
                  ("episode coverage D=5 (7-day config)", 4, 7),
                  ("episode coverage D=6 (7-day config)", 5, 7)]:
    lo, hi = wilson(k, n)
    rows.append(dict(item=tag, k=k, n=n, proportion=k / n,
                     wilson_lo=lo, wilson_hi=hi))
pd.DataFrame(rows).to_csv("P2_episode_binomial_ci.csv", index=False, encoding="utf-8-sig")


# ------------------------------------------------------------------- report
pd.set_option("display.width", 250)
print("### perfect-prog, isotonic, by subset")
for s in SUBSETS:
    print(f"\n-- {s} --")
    print(pp[(pp.subset == s) & (pp.track == "isotonic")][
        ["D", "n", "positives", "base_rate", "bss", "bss_lo", "bss_hi",
         "auc", "auc_lo", "auc_hi", "pod", "far"]].to_string(index=False,
         float_format=lambda v: f"{v:0.4f}"))

print("\n### perfect-prog, RAW, by subset")
for s in SUBSETS:
    print(f"\n-- {s} --")
    print(pp[(pp.subset == s) & (pp.track == "raw")][
        ["D", "n", "positives", "base_rate", "bss", "bss_lo", "bss_hi", "auc"]].to_string(
        index=False, float_format=lambda v: f"{v:0.4f}"))

print("\n### persistence baseline")
for s in SUBSETS:
    print(f"\n-- {s} --")
    print(pb[pb.subset == s][["D", "n", "positives", "base_rate", "n_warn",
                              "bss", "auc", "pod", "far"]].to_string(
        index=False, float_format=lambda v: f"{v:0.4f}"))

print("\n### episode coverage Wilson 95% CI")
print(pd.read_csv("P2_episode_binomial_ci.csv").to_string(index=False,
      float_format=lambda v: f"{v:0.4f}"))
