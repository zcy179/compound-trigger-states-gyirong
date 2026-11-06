# -*- coding: utf-8 -*-
"""23_gully_trigger_type.py — 84 沟触发类型分层分析。

思路来源：同事件参考竞赛论文（吴悦婷等）Problem 1 用附件(6)的
“51.2% 为非降雨型”作为独立证据链。本脚本把该字段从描述性用法升级为
正式的分层统计，供 Paper A（降雨阈值结构不足）与 Paper B（频率–规模
负相关的机理）使用。

数据：work/clean/gullies_regional.csv（含官方 附件(6) 的“类型”字段）
字段语义（官方 dbf）：
  类型 ∈ {降雨型 41, 冰水混合型 25, 冰雪融水型 16, 冰湖溃决型 2}
  规模 ∈ {小 30, 中 45, 大 7, 特大 2}
  频率 ∈ {低 21, 中 26, 高 37}

分析：
  1) 类型 × 规模 / 类型 × 频率 列联表
  2) 降雨型 vs 冰寒型：大/特大事件的 Fisher 精确检验 + 4×2 置换检验
  3) 频率–规模 Spearman：总体与分类型（置换 p）
  4) 类型间地形/物源特征差异：Kruskal–Wallis / Mann–Whitney + BH 校正
  5) 单变量 AUC：类型与冰川连通性对“大/特大”的判别力

输出：
  results/gully_trigger_type.csv
  results/gully_type_contingency.csv
  results/gully_type_tests.csv
  results/gully_type_features.csv
  results/gully_trigger_type_summary.md
  figs/fig8_trigger_type.{png,svg}
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import CLEAN, RES, FIG, C, savefig

RNG = np.random.default_rng(20260826)
NPERM = 10000

EXPECTED_TYPE = {"降雨型": 41, "冰水混合型": 25, "冰雪融水型": 16, "冰湖溃决型": 2}
TYPE_EN = {
    "降雨型": "Rainfall",
    "冰水混合型": "Ice–water mixed",
    "冰雪融水型": "Snow/ice melt",
    "冰湖溃决型": "Glacial lake outburst",
}
TYPE_ORDER = ["降雨型", "冰水混合型", "冰雪融水型", "冰湖溃决型"]
MAG = {"小": 1, "中": 2, "大": 3, "特大": 4}
FREQ = {"低": 1, "中": 2, "高": 3}
MAG_EN = {1: "Small", 2: "Medium", 3: "Large", 4: "Extreme"}
FREQ_EN = {1: "Low", 2: "Medium", 3: "High"}

g = pd.read_csv(os.path.join(CLEAN, "gullies_regional.csv"), encoding="utf-8-sig")
assert len(g) == 84, len(g)
counts = g["类型"].value_counts().to_dict()
assert counts == EXPECTED_TYPE, counts

g["type_en"] = g["类型"].map(TYPE_EN)
g["cryo"] = (g["类型"] != "降雨型").astype(int)
g["mag_ord"] = g["规模"].map(MAG)
g["freq_ord"] = g["频率"].map(FREQ)
g["large"] = (g["mag_ord"] >= 3).astype(int)


# ---------------------------------------------------------------- helpers
def bh_fdr(pvals):
    """Benjamini–Hochberg adjusted p-values (monotone)."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = min(prev, p[order[i]] * n / rank)
        q[order[i]] = val
        prev = val
    return q


def spear(x, y):
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry - ry.mean()) / ry.std()
    return float(np.mean(rx * ry))


def perm_spear(x, y, n=NPERM):
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry - ry.mean()) / ry.std()
    rho = float(np.mean(rx * ry))
    hits = 0
    for _ in range(n):
        if abs(float(np.mean(rx * RNG.permutation(ry)))) >= abs(rho) - 1e-12:
            hits += 1
    return rho, (hits + 1) / (n + 1)


def perm_chi2(groups, cats, n=NPERM):
    grp = np.asarray(groups)
    cat = np.asarray(cats)
    ug, uc = np.unique(grp), np.unique(cat)
    gi = np.searchsorted(ug, grp)
    ci = np.searchsorted(uc, cat)

    def chi2_stat(idx):
        tab = np.zeros((len(ug), len(uc)))
        np.add.at(tab, (idx, ci), 1)
        exp = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
        return float(((tab - exp) ** 2 / exp).sum())

    obs = chi2_stat(gi)
    dof = (len(ug) - 1) * (len(uc) - 1)
    p_asym = float(stats.chi2.sf(obs, dof))
    hits = sum(1 for _ in range(n)
               if chi2_stat(RNG.permutation(gi)) >= obs - 1e-12)
    return obs, dof, p_asym, (hits + 1) / (n + 1)


def auc_perm(score, y, n=NPERM):
    """Single-variable AUC and its label-permutation p-value (rank based)."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, float)
    ranks = stats.rankdata(score)
    n1 = int(y.sum())
    n0 = len(y) - n1
    hits = 0
    obs = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    for _ in range(n):
        idx = RNG.choice(len(y), size=n1, replace=False)
        auc = (ranks[idx].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
        if auc >= obs - 1e-12:
            hits += 1
    return float(obs), (hits + 1) / (n + 1)


def rank_biserial(u, n1, n2):
    """Rank-biserial correlation from Mann–Whitney U (group1 vs group2)."""
    return 2.0 * u / (n1 * n2) - 1.0


# ---------------------------------------------------------------- 1) tables
tests = []
tests.append(dict(test="type_counts", statistic=np.nan, p=np.nan, effect=np.nan,
                  n=84, note="; ".join(f"{k}={v}" for k, v in counts.items())))

# type x magnitude
mag_tab = pd.crosstab(g["类型"], g["规模"]).reindex(
    index=TYPE_ORDER, columns=["小", "中", "大", "特大"], fill_value=0)
freq_tab = pd.crosstab(g["类型"], g["频率"]).reindex(
    index=TYPE_ORDER, columns=["低", "中", "高"], fill_value=0)

# ---------------------------------------------------------------- 2) large events
rain_large = int(((g["cryo"] == 0) & (g["large"] == 1)).sum())
rain_small = int(((g["cryo"] == 0) & (g["large"] == 0)).sum())
cryo_large = int(((g["cryo"] == 1) & (g["large"] == 1)).sum())
cryo_small = int(((g["cryo"] == 1) & (g["large"] == 0)).sum())
assert (rain_large, rain_small, cryo_large, cryo_small) == (0, 41, 9, 34)

# orientation: [[cryo_large, cryo_small], [rain_large, rain_small]] -> OR>1 if cryo enriched
or_cryo, p_one = stats.fisher_exact([[cryo_large, cryo_small], [rain_large, rain_small]],
                                    alternative="greater")
_, p_two = stats.fisher_exact([[cryo_large, cryo_small], [rain_large, rain_small]],
                              alternative="two-sided")
tests.append(dict(test="fisher_large_2x2", statistic=or_cryo, p=p_one, effect=np.nan,
                  n=84, note=f"cryo {cryo_large}/{cryo_large+cryo_small} vs rain "
                             f"{rain_large}/{rain_large+rain_small}; two-sided p={p_two:.2e}"))

chi2_m, dof_m, p_asym_m, p_perm_m = perm_chi2(g["类型"], g["规模"])
cramers_m = np.sqrt(chi2_m / (84 * (min(mag_tab.shape) - 1)))
tests.append(dict(test="perm_chi2_magnitude_4x2", statistic=chi2_m, p=p_perm_m,
                  effect=cramers_m, n=84,
                  note=f"dof={dof_m}; asymptotic p={p_asym_m:.2e} (sparse)"))

chi2_f, dof_f, p_asym_f, p_perm_f = perm_chi2(g["类型"], g["频率"])
cramers_f = np.sqrt(chi2_f / (84 * (min(freq_tab.shape) - 1)))
tests.append(dict(test="perm_chi2_frequency_4x2", statistic=chi2_f, p=p_perm_f,
                  effect=cramers_f, n=84, note=f"dof={dof_f}"))

# ordinal Mann–Whitney rain vs cryo
u_m, p_m = stats.mannwhitneyu(g.loc[g["cryo"] == 1, "mag_ord"],
                              g.loc[g["cryo"] == 0, "mag_ord"],
                              alternative="two-sided")
tests.append(dict(test="mwu_magnitude_rain_vs_cryo", statistic=u_m, p=p_m,
                  effect=rank_biserial(u_m, 43, 41), n=84,
                  note="ordinal scale 1–4; positive effect = cryo larger"))
u_f, p_f = stats.mannwhitneyu(g.loc[g["cryo"] == 1, "freq_ord"],
                              g.loc[g["cryo"] == 0, "freq_ord"],
                              alternative="two-sided")
tests.append(dict(test="mwu_frequency_rain_vs_cryo", statistic=u_f, p=p_f,
                  effect=rank_biserial(u_f, 43, 41), n=84,
                  note="ordinal scale 1–3; positive effect = cryo higher frequency"))

# ---------------------------------------------------------------- 3) frequency-magnitude
rho_all, p_all = perm_spear(g["freq_ord"].values, g["mag_ord"].values)
tests.append(dict(test="spearman_freq_mag_overall", statistic=rho_all, p=p_all,
                  effect=np.nan, n=84, note="permutation p"))
for t in TYPE_ORDER:
    sub = g[g["类型"] == t]
    if len(sub) >= 5:
        rho, p = perm_spear(sub["freq_ord"].values, sub["mag_ord"].values)
        tests.append(dict(test=f"spearman_freq_mag_{TYPE_EN[t]}", statistic=rho, p=p,
                          effect=np.nan, n=len(sub), note="permutation p"))
for name, mask in [("rain_vs_cryo", g["cryo"] == 0),
                   ("cryo_combined", g["cryo"] == 1)]:
    sub = g[mask]
    rho, p = perm_spear(sub["freq_ord"].values, sub["mag_ord"].values)
    tests.append(dict(test=f"spearman_freq_mag_{name}", statistic=rho, p=p,
                      effect=np.nan, n=len(sub), note="permutation p"))

# ---------------------------------------------------------------- 4) single-variable AUCs
for col, sign, label in [
    ("cryo", 1, "cryo_flag"),
    ("glacier_frac_mean", 1, "glacier_frac"),
    ("lc10", 1, "lc10_snow_ice"),
    ("dist_glacier_mean", -1, "dist_to_glacier"),
    ("height", -1, "fan_elevation"),
]:
    score = sign * g[col].astype(float).values
    auc, p_auc = auc_perm(score, g["large"].values)
    tests.append(dict(test=f"auc_large_vs_{label}", statistic=auc, p=p_auc,
                      effect=np.nan, n=84,
                      note="single-variable AUC for large/extreme (>=3)"))
# also AUC for high frequency
for col, sign, label in [("cryo", -1, "cryo_flag"), ("glacier_frac_mean", -1, "glacier_frac"),
                         ("dist_glacier_mean", 1, "dist_to_glacier")]:
    score = sign * g[col].astype(float).values
    y = (g["freq_ord"] == 3).astype(int).values
    auc, p_auc = auc_perm(score, y)
    tests.append(dict(test=f"auc_highfreq_vs_{label}", statistic=auc, p=p_auc,
                      effect=np.nan, n=84,
                      note="single-variable AUC for high frequency class"))

# ---------------------------------------------------------------- 5) continuous features
FEATURES = [
    ("height", "Fan elevation (m)"),
    ("高程Max", "Maximum elevation (m)"),
    ("高差", "Catchment relief (m)"),
    ("沟比降", "Channel slope (-)"),
    ("主沟长", "Main channel length (m)"),
    ("Area", "Catchment area (km2)"),
    ("glacier_frac_mean", "Glacier fraction (-)"),
    ("lc10", "Snow/ice land-cover fraction (-)"),
    ("lc7", "Bare/till land-cover fraction (-)"),
    ("slope_mean", "Mean slope (deg)"),
    ("dist_glacier_mean", "Distance to glacier (m)"),
]
main3 = [t for t in TYPE_ORDER if (g["类型"] == t).sum() >= 5]
feat_rows = []
for col, label in FEATURES:
    vals3 = [g.loc[g["类型"] == t, col].astype(float).values for t in main3]
    H, p_kw = stats.kruskal(*vals3)
    n_kw = sum(len(v) for v in vals3)
    eps2 = (H - len(vals3) + 1) / (n_kw - len(vals3)) if n_kw > len(vals3) else np.nan
    u, p_mw = stats.mannwhitneyu(g.loc[g["cryo"] == 1, col].astype(float),
                                 g.loc[g["cryo"] == 0, col].astype(float),
                                 alternative="two-sided")
    rb = rank_biserial(u, 43, 41)
    for t in TYPE_ORDER:
        v = g.loc[g["类型"] == t, col].astype(float)
        feat_rows.append(dict(feature=col, feature_en=label, group=TYPE_EN[t],
                              n=len(v), median=v.median(), q1=v.quantile(.25),
                              q3=v.quantile(.75), mean=v.mean(),
                              kw_H=H, kw_p=p_kw, kw_epsilon2=eps2,
                              mwu_p_rain_vs_cryo=p_mw, rank_biserial_cryo=rb))
feat = pd.DataFrame(feat_rows)
q_bh = bh_fdr(feat.drop_duplicates("feature")["kw_p"].values)
feat = feat.merge(pd.DataFrame({"feature": feat.drop_duplicates("feature")["feature"].values,
                                "kw_p_bh": q_bh}), on="feature", how="left")

# append KW rows to tests
n_kw = int(sum((g["类型"] == t).sum() for t in main3))
for _, row in feat.drop_duplicates("feature").iterrows():
    tests.append(dict(test=f"kruskal_{row['feature']}", statistic=row["kw_H"],
                      p=row["kw_p"], effect=row["kw_epsilon2"], n=int(n_kw),
                      note=f"3 main types (n={n_kw}); BH q={row['kw_p_bh']:.3g}; "
                           f"MWU rain-vs-cryo p={row['mwu_p_rain_vs_cryo']:.3g}"))
tests_df = pd.DataFrame(tests)

# ---------------------------------------------------------------- outputs
per_gully = g[["Code", "x_coord", "y_coord", "height", "高程Max", "高差", "沟比降",
               "主沟长", "Area", "规模", "频率", "类型", "type_en", "cryo",
               "mag_ord", "freq_ord", "large", "glacier_frac_mean", "lc7", "lc10",
               "slope_mean", "acc_mean", "hand_mean", "dist_glacier_mean",
               "area_fan_km2"]].copy()
per_gully = per_gully.sort_values(["type_en", "Code"]).reset_index(drop=True)
per_gully.to_csv(os.path.join(RES, "gully_trigger_type.csv"), index=False, encoding="utf-8-sig")

cont_rows = []
for tab, tabname, order in [(mag_tab, "magnitude", ["小", "中", "大", "特大"]),
                            (freq_tab, "frequency", ["低", "中", "高"])]:
    for t in TYPE_ORDER:
        row_sum = int(tab.loc[t].sum())
        for c in order:
            n = int(tab.loc[t, c])
            cont_rows.append(dict(table=tabname, type_cn=t, type_en=TYPE_EN[t],
                                  class_cn=c, n=n, row_n=row_sum,
                                  row_pct=100.0 * n / row_sum if row_sum else np.nan))
pd.DataFrame(cont_rows).to_csv(os.path.join(RES, "gully_type_contingency.csv"),
                               index=False, encoding="utf-8-sig")
tests_df.to_csv(os.path.join(RES, "gully_type_tests.csv"), index=False, encoding="utf-8-sig")
feat.to_csv(os.path.join(RES, "gully_type_features.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- figure
plt.rcParams["font.family"] = "DejaVu Sans"
SHORT = {"降雨型": "Rainfall", "冰水混合型": "Ice–water",
         "冰雪融水型": "Melt", "冰湖溃决型": "GLOF"}
fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))

# (a) magnitude composition
ax = axes[0]
mag_colors = [C["lightblue"], C["teal"], C["orange"], C["red"]]
bottoms = np.zeros(len(TYPE_ORDER))
for j, c in enumerate(["小", "中", "大", "特大"]):
    vals = np.array([100.0 * mag_tab.loc[t, c] / mag_tab.loc[t].sum() for t in TYPE_ORDER])
    ax.bar(range(len(TYPE_ORDER)), vals, bottom=bottoms, color=mag_colors[j],
           edgecolor="white", linewidth=0.6, label=MAG_EN[j + 1])
    bottoms += vals
ax.set_xticks(range(len(TYPE_ORDER)))
ax.set_xticklabels([f"{SHORT[t]}\n(n={mag_tab.loc[t].sum()})" for t in TYPE_ORDER],
                   fontsize=7.8)
ax.set_ylabel("Magnitude class (%)")
ax.set_ylim(0, 100)
ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
ax.set_title("(a) Event magnitude by trigger type", fontsize=9)

# (b) frequency composition
ax = axes[1]
freq_colors = [C["grey"], C["yellow"], C["blue"]]
bottoms = np.zeros(len(TYPE_ORDER))
for j, c in enumerate(["低", "中", "高"]):
    vals = np.array([100.0 * freq_tab.loc[t, c] / freq_tab.loc[t].sum() for t in TYPE_ORDER])
    ax.bar(range(len(TYPE_ORDER)), vals, bottom=bottoms, color=freq_colors[j],
           edgecolor="white", linewidth=0.6, label=FREQ_EN[j + 1])
    bottoms += vals
ax.set_xticks(range(len(TYPE_ORDER)))
ax.set_xticklabels([f"{SHORT[t]}\n(n={freq_tab.loc[t].sum()})" for t in TYPE_ORDER],
                   fontsize=7.8)
ax.set_ylabel("Frequency class (%)")
ax.set_ylim(0, 100)
ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
ax.set_title("(b) Event frequency by trigger type", fontsize=9)

# (c) frequency vs magnitude jitter
ax = axes[2]
rng = np.random.default_rng(7)
type_colors = {"降雨型": C["blue"], "冰水混合型": C["teal"],
               "冰雪融水型": C["orange"], "冰湖溃决型": C["purple"]}
markers = {"降雨型": "o", "冰水混合型": "s", "冰雪融水型": "^", "冰湖溃决型": "D"}
for t in TYPE_ORDER:
    sub = g[g["类型"] == t]
    jx = sub["freq_ord"] + rng.uniform(-0.16, 0.16, len(sub))
    jy = sub["mag_ord"] + rng.uniform(-0.14, 0.14, len(sub))
    ax.scatter(jx, jy, s=22, color=type_colors[t], marker=markers[t],
               alpha=0.85, edgecolor="white", linewidth=0.35, label=TYPE_EN[t])
ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["Low", "Medium", "High"], fontsize=8)
ax.set_yticks([1, 2, 3, 4]); ax.set_yticklabels(["Small", "Medium", "Large", "Extreme"], fontsize=8)
ax.set_xlabel("Historical frequency class"); ax.set_ylabel("Magnitude class")
ax.set_title(f"(c) Spearman ρ = {rho_all:.2f} (overall)", fontsize=9)
ax.legend(fontsize=6.5, loc="upper left", frameon=True, framealpha=0.85,
          edgecolor="0.8", borderpad=0.3)

fig.tight_layout()
savefig(fig, "fig8_trigger_type")

# ---------------------------------------------------------------- summary md
def pct(n, d):
    return f"{100.0*n/d:.1f}%"


lines = []
lines.append("# 84 沟触发类型分层分析（供 Paper A / B 使用）\n")
lines.append(f"- 脚本：`work/code/23_gully_trigger_type.py`")
lines.append(f"- 逐沟表：`work/results/gully_trigger_type.csv`（84 行，含官方 `类型` 字段）")
lines.append(f"- 列联表：`work/results/gully_type_contingency.csv`")
lines.append(f"- 检验：`work/results/gully_type_tests.csv`")
lines.append(f"- 特征：`work/results/gully_type_features.csv`")
lines.append(f"- 图：`work/figs/fig8_trigger_type.svg`\n")
lines.append("> 数据源：`work/clean/gullies_regional.csv`，其 `类型` 字段来自官方附件(6) dbf；"
             "本脚本已断言类型计数为 41/25/16/2。\n")
lines.append("> **解释边界**：`类型` 是灾后触发机制分类，不是事前可观测变量。"
             "它用于分层诊断；若要做事前制图，应改用冰川覆盖率、雪冰土地覆盖、到冰川距离等代理。\n")

lines.append("## 1. 类型 × 规模\n")
lines.append("| 类型 | 小 | 中 | 大 | 特大 | n | 大+特大占比 |")
lines.append("|---|---|---|---|---|---|---|")
for t in TYPE_ORDER:
    row = [int(mag_tab.loc[t, c]) for c in ["小", "中", "大", "特大"]]
    lines.append(f"| {TYPE_EN[t]} | {row[0]} | {row[1]} | {row[2]} | {row[3]} | "
                 f"{sum(row)} | {pct(row[2]+row[3], sum(row))} |")
lines.append("")

lines.append("## 2. 类型 × 频率\n")
lines.append("| 类型 | 低 | 中 | 高 | n | 高频率占比 |")
lines.append("|---|---|---|---|---|---|")
for t in TYPE_ORDER:
    row = [int(freq_tab.loc[t, c]) for c in ["低", "中", "高"]]
    lines.append(f"| {TYPE_EN[t]} | {row[0]} | {row[1]} | {row[2]} | "
                 f"{sum(row)} | {pct(row[2], sum(row))} |")
lines.append("")

lines.append("## 3. 关键检验\n")
sel = tests_df[tests_df.test.str.startswith((
    "fisher_large", "perm_chi2", "mwu_", "spearman_freq_mag", "auc_large", "auc_highfreq"))]
for _, r in sel.iterrows():
    p = r["p"]
    pstr = "n/a" if pd.isna(p) else (f"{p:.2e}" if p < 1e-3 else f"{p:.3f}")
    lines.append(f"- `{r['test']}`: statistic = {r['statistic']:.3g}, p = {pstr}, "
                 f"effect = {r['effect']:.3g}" if not pd.isna(r["effect"]) else
                 f"- `{r['test']}`: statistic = {r['statistic']:.3g}, p = {pstr}")
lines.append("")

lines.append("## 4. 特征中位数（按类型）\n")
lines.append("> Kruskal–Wallis 用三个主类型（降雨/冰水混合/冰雪融水，n=82）；"
             "冰湖溃决型仅 2 条，仅在表中列出中位数，不参与检验。\n")
key = ["height", "高差", "沟比降", "glacier_frac_mean", "lc10", "lc7", "dist_glacier_mean"]
lines.append("| 特征 | " + " | ".join(TYPE_EN[t] for t in TYPE_ORDER) + " | KW p (BH) |")
lines.append("|---|" + "---|" * len(TYPE_ORDER) + "---|")
for col in key:
    sub = feat[feat.feature == col]
    cells = []
    for t in TYPE_ORDER:
        rr = sub[sub.group == TYPE_EN[t]].iloc[0]
        cells.append(f"{rr['median']:.3g} [{rr['q1']:.3g}, {rr['q3']:.3g}]")
    rr0 = sub.iloc[0]
    lines.append(f"| {rr0['feature_en']} | " + " | ".join(cells) +
                 f" | {rr0['kw_p']:.2e} ({rr0['kw_p_bh']:.2e}) |")
lines.append("")

lines.append("## 5. 可直接写进论文的结论\n")
lines.append(f"1. **大/特大事件全部来自冰寒型沟谷**：降雨型 {rain_large}/{rain_large+rain_small}，"
             f"冰寒型 {cryo_large}/{cryo_large+cryo_small}；Fisher 单侧 p = {p_one:.2e}。"
             f"这为 Paper A “降雨单变量阈值在结构上覆盖不了最大事件”提供独立于 2026 事件的证据。")
lines.append(f"2. **类型与规模显著相关**：4×2 置换检验 χ² = {chi2_m:.2f}, p = {p_perm_m:.2e}, "
             f"Cramér's V = {cramers_m:.2f}；rain-vs-cryo 有序 MWU p = {p_m:.2e}。")
lines.append(f"3. **频率–规模负相关不是单一总体现象**：总体 ρ = {rho_all:.2f} (p = {p_all:.3f})；"
             f"降雨型 ρ = {tests_df.loc[tests_df.test=='spearman_freq_mag_Rainfall','statistic'].iloc[0]:.2f} "
             f"(p = {tests_df.loc[tests_df.test=='spearman_freq_mag_Rainfall','p'].iloc[0]:.3f})，"
             f"冰雪融水型 ρ = {tests_df.loc[tests_df.test=='spearman_freq_mag_Snow/ice melt','statistic'].iloc[0]:.2f} "
             f"(p = {tests_df.loc[tests_df.test=='spearman_freq_mag_Snow/ice melt','p'].iloc[0]:.3f})，"
             f"冰水混合型 ρ = {tests_df.loc[tests_df.test=='spearman_freq_mag_Ice–water mixed','statistic'].iloc[0]:.2f} "
             f"(p = {tests_df.loc[tests_df.test=='spearman_freq_mag_Ice–water mixed','p'].iloc[0]:.3f})。"
             f"负相关主要由降雨型驱动，融水型边缘支持，冰水混合型内部不存在；"
             f"Paper B 的“复合指数失败”应改写为“不同触发类型总体的混合”，而不是单一连续体的权衡。")
lines.append(f"4. **类型标签提供中等强度的单变量判别力**：AUC(cryo → large/extreme) = "
             f"{tests_df.loc[tests_df.test=='auc_large_vs_cryo_flag','statistic'].iloc[0]:.3f} "
             f"(p = {tests_df.loc[tests_df.test=='auc_large_vs_cryo_flag','p'].iloc[0]:.4f})；"
             f"Paper B 的 RF 幅度模型 AUC 0.847 高于该值，说明地形/土地覆盖特征在类型之外仍有增量，"
             f"但“冰寒型 vs 降雨型”的对比解释了相当一部分技巧。注意类型是事后分类，"
             f"业务上应改用可事前观测的代理：冰川覆盖率、雪冰土地覆盖、到冰川距离。")
lines.append(f"5. **频率与类型的关系只是边缘显著**：AUC(cryo → high freq) = "
             f"{tests_df.loc[tests_df.test=='auc_highfreq_vs_cryo_flag','statistic'].iloc[0]:.3f} "
             f"(p = {tests_df.loc[tests_df.test=='auc_highfreq_vs_cryo_flag','p'].iloc[0]:.4f})，"
             f"4×2 频率检验 p = {p_perm_f:.3f}；降雨型的高频率占比更高（53.7% vs 冰水混合型 28.0%），"
             f"但不能说“高频率=降雨型”。结论应限定为：频率与幅度描述不同总体，复合相乘仍不成立。")
lines.append("")
lines.append("## 6. 已插入论文的位置（2026-10-30）\n")
lines.append("> **独立投稿约束**：A、B 两篇按互不依赖的独立稿件处理，正文与参考文献不得出现"
             "“companion paper / Paper A / Paper B”式交叉引用；共享内容只保留事件事实、数据来源与各自的必要定义，"
             "分析不重复——A 只保留 2×2 Fisher 检验，B 保留完整分层分析。\n")
lines.append("**Paper A**")
lines.append("- §2.5（新增小节）“Independent regional trigger-type evidence”：数据来源、类型/规模计数、"
             "“类型只作分层变量、不作预报输入”的边界，以及引用 Supplementary Table S1。")
lines.append("- §4.1 末段：只保留 2×2 Fisher 检验（0/41 vs 9/43，p = 1.5e-3），并声明这是对“复合指标前提”的检验，"
             "不是对指标的验证；有序 MWU、4×2 χ²、频率检验等只出现在 B 篇，避免两稿重复。")
lines.append("- §5.6 第二段：I–D 阈值以降雨型滑坡为标定总体，不能直接外推到高寒链式灾害；"
             "在本文走廊里大事件来自冰寒型。")
lines.append("- Table 1 增行（84 沟清单）；References 增 TPDC (2026)；Supplementary 说明增 Table S6"
             "（补充材料 Table S6 同步替换为 2×2 表 + Fisher 检验）；Data/code 增 `23_gully_trigger_type.py`。")
lines.append("")
lines.append("**Paper B**")
lines.append("- §3.6 末段：类型分层的统计设计（置换 χ²、MWU/rank-biserial、AUC + 置换 p、KW + BH、"
             "GLOF 不参与检验）。")
lines.append("- §4.5 新增三段：类型—规模关联 + 代理变量 AUC；频率—规模负相关按类型分解（混合效应）；"
             "分层结果自身的样本量限制。")
lines.append("- §5.5 新增一段：复合指数实质是“单一总体模型”；建议先定触发体制、再在体制内合成。")
lines.append("- Abstract、§1 第五点、Conclusions 第 6 条、§5.6 限制第 9 条：补充分层结论与边界。")
lines.append("- Table 3 增“trigger-type classes”；新增 **Table 11**（列联表 A/B + 检验 C）与 **Fig. 8**；"
             "Data/code 增 `23_gully_trigger_type.py`。")
lines.append("")
lines.append("> 未借用的参考文内容：其 51.2% 描述性统计及其结论、CRS/P25–P75 阈值、"
             "“事件日水源未触发”的表述均未进入论文；分层统计由本脚本独立重算。")
lines.append("")
with open(os.path.join(RES, "gully_trigger_type_summary.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("OK: wrote gully_trigger_type.csv / contingency / tests / features / summary.md / fig8")
print("large events: rain %d/%d, cryo %d/%d, Fisher one-sided p=%.3e"
      % (rain_large, rain_large + rain_small, cryo_large, cryo_large + cryo_small, p_one))
print("overall rho=%.3f (p_perm=%.4f)" % (rho_all, p_all))
