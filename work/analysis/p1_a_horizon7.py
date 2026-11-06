# -*- coding: utf-8 -*-
"""P1-A6/A9: seven-day-horizon reanalysis of the CTI first-passage forecast.

Why: the stored 3-day hindcast (p3_hindcast_cases.csv) cannot score an event
six days after the issue date.  The event forecast (p3_event_forecast.csv)
also used H=3, so "122.5 h" is a red-alert-to-event interval, not a probability
that 26 August crosses C*.  This script re-runs the same Gaussian-copula CTI
engine with H=7 and a calibration window that stops on 2026-08-19 (event day
excluded), and reports event-specific probabilities.

Outputs (work/analysis/):
  P1_A6_h7_hindcast_cases.csv
  P1_A6_h7_leadtime.csv
  P1_A6_h7_conformal.csv
  P1_A6_h7_event_forecast.csv
  P1_A6_h7_summary.md
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
CLEAN = os.path.join(WORK, "clean")
DATA = os.path.join(WORK, "data")
RES = os.path.join(WORK, "results")
OUT = HERE

CSTAR = 2.5
P_STAR = 0.05
HMAX = 7
N_HIND = 1500
N_ENS = 4000
RNG_H = np.random.default_rng(20260826)      # hindcast engine
RNG_E = np.random.default_rng(20260826)      # event engine
CUTOFF = pd.Timestamp("2026-08-19")          # event excluded from all fits
EVT = pd.Timestamp("2026-08-26")
EVT_T = pd.Timestamp("2026-08-26 10:52")     # USGS 02:52:10 UTC

# ------------------------------------------------------------------ data
d = pd.read_csv(os.path.join(CLEAN, "daily.csv"), parse_dates=["date"]).sort_values("date")
d["API7"] = d["API"].rolling(7, min_periods=5).mean()
d["PDD_1d"] = np.maximum(d["Ta_2_mean"], 0.0)
COLS = ["API7", "PDD_3d", "VPD_kPa_max"]
SEAS = d.date.dt.month.between(5, 10)
TRAIN = d[SEAS & (d.date < "2026-01-01")].dropna(subset=COLS).reset_index(drop=True)
N = len(TRAIN)
U = np.column_stack([stats.rankdata(TRAIN[c], method="average") / (N + 1) for c in COLS])
TAU = np.array([stats.kendalltau(U[:, i], U[:, j])[0] for i, j in [(0, 1), (0, 2), (1, 2)]])
SIG = np.array([[1, np.sin(np.pi * TAU[0] / 2), np.sin(np.pi * TAU[1] / 2)],
                [np.sin(np.pi * TAU[0] / 2), 1, np.sin(np.pi * TAU[2] / 2)],
                [np.sin(np.pi * TAU[1] / 2), np.sin(np.pi * TAU[2] / 2), 1]])
SIG = (SIG + SIG.T) / 2
MARG = {c: np.sort(TRAIN[c].values) for c in COLS}
_XG, _WG = np.polynomial.legendre.leggauss(40)
_TG, _WT = np.polynomial.legendre.leggauss(20)


def p3_joint(z):
    z = np.atleast_2d(z)
    z1, z2, z3 = z[:, 0], z[:, 1], z[:, 2]
    xs = z1[:, None] + 4.0 * (1 + _XG)[None, :]
    wxs = 4.0 * _WG[None, :] * stats.norm.pdf(xs)
    r12, r13, r23 = SIG[0, 1], SIG[0, 2], SIG[1, 2]
    s2, s3 = np.sqrt(1 - r12 ** 2), np.sqrt(1 - r13 ** 2)
    rc = (r23 - r12 * r13) / (s2 * s3)
    a = (z2[:, None] - r12 * xs) / s2
    b = (z3[:, None] - r13 * xs) / s3
    pa, pb = stats.norm.cdf(a), stats.norm.cdf(b)
    Phi2 = pa * pb
    for tk, wk in zip(_TG, _WT):
        t = 0.5 * rc * (tk + 1.0)
        Phi2 += (0.5 * rc * wk * np.exp(-(a * a + b * b - 2 * a * b * t) /
                                        (2 * (1 - t * t))) /
                 (2 * np.pi * np.sqrt(1 - t * t)))
    return np.clip(((1 - pa - pb + Phi2) * wxs).sum(1), 1e-15, 1.0)


def ecdf_v(s, x):
    s = np.sort(np.asarray(s, float)); x = np.asarray(x, float)
    return np.clip((np.searchsorted(s, x, side="right") + 0.5) / (len(s) + 1),
                   0.5 / (len(s) + 1), 1 - 0.5 / (len(s) + 1))


def cti_from(api7, pdd3, vpd):
    sh = np.shape(api7)
    u1 = ecdf_v(MARG["API7"], api7); u2 = ecdf_v(MARG["PDD_3d"], pdd3)
    u3 = ecdf_v(MARG["VPD_kPa_max"], vpd)
    z = np.column_stack([stats.norm.ppf(u1).ravel(), stats.norm.ppf(u2).ravel(),
                         stats.norm.ppf(u3).ravel()])
    return (-np.log10(p3_joint(z))).reshape(sh)


d["CTI"] = cti_from(d["API7"].values, d["PDD_3d"].values, d["VPD_kPa_max"].values)
obs = d.dropna(subset=COLS + ["CTI"]).reset_index(drop=True)

# ------------------------------------------------------------------ NWP calibration
fc = pd.read_csv(os.path.join(DATA, "meteo", "fcst_port.csv"), parse_dates=["time"])
fc["date"] = fc["time"].dt.normalize()


def esat(t):
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


fc["VPD"] = (esat(fc["temperature_2m"]) * (1 - fc["relative_humidity_2m"] / 100)).clip(lower=0)
fd = fc.groupby("date").agg(Tmax_raw=("temperature_2m", "max"),
                            Tmean_raw=("temperature_2m", "mean"),
                            VPD_raw=("VPD", "max"), P=("precipitation", "sum"),
                            PS=("surface_pressure", "median")).reset_index()
st_cols = d[["date", "Ta_2_max", "Ta_2_mean", "VPD_kPa_max", "Precip_sum", "API"]]
_m = fd.merge(st_cols, on="date", how="inner")
_m = _m[_m.date <= CUTOFF]
Z_ST = 4140.0                                  # P0-9 adjudicated station elevation
z_nwp = 44330 * (1 - (_m["PS"].median() / 1013.25) ** 0.1903)
GAM = -float(np.mean(_m["Ta_2_max"] - _m["Tmax_raw"])) / (Z_ST - z_nwp)
PFAC = float(_m["Precip_sum"].sum() / max(_m["P"].sum(), 1e-6))
fd["Tmean_cal"] = fd["Tmean_raw"] - GAM * (Z_ST - z_nwp)
fd["Tmax_cal"] = fd["Tmax_raw"] - GAM * (Z_ST - z_nwp)
fd["P_cal"] = fd["P"] * PFAC
_sx, _sy = np.sort(_m["VPD_raw"].values), np.sort(_m["VPD_kPa_max"].values)
fd["VPD_qm"] = np.interp(fd["VPD_raw"].values, _sx, _sy)
cmp = fd.merge(st_cols, on="date", how="inner")
cmp = cmp[cmp.date <= CUTOFF]
TM_BIAS = float(np.mean(cmp["Tmean_cal"] - cmp["Ta_2_mean"]))
fd["Tmean_cal"] = fd["Tmean_cal"] - TM_BIAS
cmp = fd.merge(st_cols, on="date", how="inner")
cmp = cmp[cmp.date <= CUTOFF]


def acf1(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


rT = cmp["Tmean_cal"] - cmp["Ta_2_mean"]
rV = cmp["VPD_qm"] - cmp["VPD_kPa_max"]
SIG_T = float(np.std(rT, ddof=1)); RHO_T = abs(acf1(rT))
SIG_V = float(np.std(rV, ddof=1)); RHO_V = abs(acf1(rV))
_p3o = cmp["Precip_sum"].rolling(3, min_periods=1).sum()
_p3f = cmp["P_cal"].rolling(3, min_periods=1).sum()
SIG_P = float(np.std(np.log((_p3f + 1) / (_p3o + 1))))
print("calibration cutoff %s: n=%d GAM=%.3f K/km PFAC=%.4f TM_BIAS=%.3f "
      "SIG_T=%.3f RHO_T=%.3f SIG_V=%.3f SIG_P=%.2f"
      % (CUTOFF.date(), len(cmp), GAM * 1000, PFAC, TM_BIAS, SIG_T, RHO_T, SIG_V, SIG_P))

fd = fd.sort_values("date").reset_index(drop=True)
fd["API"] = np.nan
api = float(d.loc[d.date == "2026-06-30", "API"].iloc[0])
for i in range(len(fd)):
    api = 0.9 * api + fd.loc[i, "P_cal"]
    fd.loc[i, "API"] = api
fd["API7"] = fd["API"].rolling(7, min_periods=5).mean()
fd["PDD"] = np.maximum(fd["Tmean_cal"], 0.0)
fd["PDD_3d"] = fd["PDD"].rolling(3, min_periods=1).sum()
fd["CTI"] = cti_from(fd["API7"].values, fd["PDD_3d"].values, fd["VPD_qm"].values)

# ------------------------------------------------------------------ engine
def make_state(t0):
    i = obs.index[obs.date == t0][0]
    return dict(A=float(obs.loc[i - 1, "API"]),
                api_hist=obs.loc[i - 6:i - 1, "API"].values.astype(float),
                pdd_hist=obs.loc[i - 2:i - 1, "PDD_1d"].values.astype(float))


def run_ensemble(t0, fc_T, fc_V, fc_P, state, n, rng):
    H = len(fc_T)
    zT = rng.standard_normal((n, H)); zV = rng.standard_normal((n, H))
    epsT = np.zeros((n, H)); epsV = np.zeros((n, H))
    growth = np.sqrt(1 + 0.75 * np.arange(H))
    for k in range(H):
        if k == 0:
            epsT[:, 0] = SIG_T * growth[0] * zT[:, 0]
            epsV[:, 0] = SIG_V * growth[0] * zV[:, 0]
        else:
            epsT[:, k] = RHO_T * epsT[:, k - 1] + SIG_T * growth[k] * np.sqrt(1 - RHO_T ** 2) * zT[:, k]
            epsV[:, k] = RHO_V * epsV[:, k - 1] + SIG_V * growth[k] * np.sqrt(1 - RHO_V ** 2) * zV[:, k]
    pdd_day = np.maximum(np.asarray(fc_T, float)[None, :] + epsT, 0.0)
    mult = np.exp(rng.normal(-0.5 * SIG_P ** 2, SIG_P, (n, H)))
    Pen = np.asarray(fc_P, float)[None, :] * mult
    A = np.zeros((n, H)); api = np.full(n, state["A"])
    for k in range(H):
        api = 0.9 * api + Pen[:, k]; A[:, k] = api
    B = np.concatenate([np.tile(state["api_hist"], (n, 1)), A], axis=1)
    api7 = np.column_stack([B[:, k:k + 7].mean(1) for k in range(H)])
    ph = np.concatenate([np.tile(state["pdd_hist"], (n, 1)), pdd_day], axis=1)
    pdd3 = np.column_stack([ph[:, k:k + 3].sum(1) for k in range(H)])
    V = np.clip(np.asarray(fc_V, float)[None, :] + epsV, 1e-3, None)
    return cti_from(api7, pdd3, V)


def onset_stats(CTI, c=CSTAR):
    red = CTI >= c
    has = red.any(1)
    H = CTI.shape[1]
    out = dict(p_day=red.mean(0), p_any=float(has.mean()),
               p24=float(red[:, 0].mean()), p48=float(red[:, :2].any(1).mean()),
               p72=float(red[:, :3].any(1).mean()))
    if has.sum() == 0:
        out.update(onset_med=np.nan, onset_lo=np.nan, onset_hi=np.nan)
        return out
    onset = red.argmax(1)[has]
    out.update(onset_med=float(np.median(onset)),
               onset_lo=float(np.percentile(onset, 2.5)),
               onset_hi=float(np.percentile(onset, 97.5)))
    return out


# ------------------------------------------------------------------ H=7 hindcast
hm = obs[obs.date.dt.month.between(5, 10)].reset_index(drop=True)
hm["is_red"] = hm["CTI"] >= CSTAR
rows = []
for i in range(3, len(hm) - HMAX + 1):
    t0 = hm.loc[i, "date"]
    if t0 > pd.Timestamp("2026-07-31"):
        continue
    if hm.loc[i - 1, "CTI"] >= CSTAR:
        continue
    sub = hm.iloc[i:i + HMAX]
    if len(sub) < HMAX or sub[COLS].isna().any().any():
        continue
    st = make_state(t0)
    CTI = run_ensemble(t0, sub["Ta_2_mean"].values, sub["VPD_kPa_max"].values,
                       sub["Precip_sum"].values, st, N_HIND, RNG_H)
    s = onset_stats(CTI)
    any_obs = bool((sub["CTI"] >= CSTAR).any())
    ons_obs = int(np.argmax((sub["CTI"] >= CSTAR).values)) if any_obs else -1
    row = dict(issue=t0, year=t0.year, any_obs=int(any_obs), ons_obs=ons_obs,
               onset_med=s["onset_med"], onset_lo=s["onset_lo"], onset_hi=s["onset_hi"])
    for D in range(1, HMAX + 1):
        row[f"p_d{D}"] = float((CTI[:, :D] >= CSTAR).any(1).mean())
        row[f"y_d{D}"] = int(bool((sub["CTI"].iloc[:D] >= CSTAR).any()))
    rows.append(row)
hd = pd.DataFrame(rows)
hd["p_any"] = hd["p_d7"]
print("H=7 hindcast: n=%d positives=%d (%.1f%%)"
      % (len(hd), int(hd.any_obs.sum()), 100 * hd.any_obs.mean()))

# isotonic per horizon, train 2021-2023, test 2024-2026
iso = {}
for D in range(1, HMAX + 1):
    m = hd.year <= 2023
    iso[D] = IsotonicRegression(out_of_bounds="clip").fit(hd.loc[m, f"p_d{D}"], hd.loc[m, f"y_d{D}"])
    hd[f"pc_d{D}"] = iso[D].predict(hd[f"p_d{D}"].values)
hd["p_cal"] = hd["pc_d7"]

lead_rows = []
for D in range(1, HMAX + 1):
    y = hd[f"y_d{D}"].values.astype(int)
    for tag, pcol in [("raw", f"p_d{D}"), ("calibrated", f"pc_d{D}")]:
        p = hd[pcol].values.astype(float)
        base = y.mean(); brier = np.mean((p - y) ** 2)
        bss = 1 - brier / (base * (1 - base)) if 0 < base < 1 else np.nan
        auc = roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan
        w = p >= P_STAR
        tp = int((w & (y == 1)).sum()); fp = int((w & (y == 0)).sum())
        fn = int((~w & (y == 1)).sum())
        lead_rows.append(dict(horizon=D, track=tag, n=len(y), pos=int(y.sum()),
                              base=base, brier=brier, bss=bss, auc=auc,
                              pod=tp / max(tp + fn, 1), far=fp / max(tp + fp, 1),
                              csi=tp / max(tp + fp + fn, 1), nwarn=int(w.sum())))
lt = pd.DataFrame(lead_rows)
print("\nH=7 perfect-prog lead-time skill\n", lt.round(3).to_string(index=False))

# moving-block bootstrap (10-day blocks within each season), fixed climatology
RNG_B = np.random.default_rng(20261029)


def block_ci(issue, y, p, nboot=1000, block=10):
    df = pd.DataFrame({"issue": pd.to_datetime(issue), "y": np.asarray(y, int),
                       "p": np.asarray(p, float)})
    df["block"] = df.issue.dt.year * 100 + (df.issue.dt.dayofyear // block)
    blocks = [g for _, g in df.groupby("block")]
    base = df.y.mean(); bs, ba = [], []
    for _ in range(nboot):
        s = pd.concat([blocks[i] for i in RNG_B.integers(0, len(blocks), len(blocks))],
                      ignore_index=True)
        yy, pp = s.y.values, s.p.values
        if yy.min() == yy.max():
            continue
        bs.append(1 - np.mean((pp - yy) ** 2) / max(base * (1 - base), 1e-9))
        ba.append(roc_auc_score(yy, pp))
    return np.percentile(bs, [2.5, 97.5]), np.percentile(ba, [2.5, 97.5])


boot_rows = []
for D in range(1, HMAX + 1):
    y = hd[f"y_d{D}"].values.astype(int)
    for tag, pcol in [("raw", f"p_d{D}"), ("calibrated", f"pc_d{D}")]:
        bl, al = block_ci(hd.issue, y, hd[pcol].values)
        r = lt[(lt.horizon == D) & (lt.track == tag)].iloc[0]
        boot_rows.append(dict(horizon=D, track=tag, bss=r.bss, bss_lo=bl[0], bss_hi=bl[1],
                              auc=r.auc, auc_lo=al[0], auc_hi=al[1]))
boot = pd.DataFrame(boot_rows)
print("\nH=7 block-bootstrap CIs\n", boot.round(3).to_string(index=False))

# conformal per horizon (positives only)
cov_rows = []
for D in range(1, HMAX + 1):
    pos = hd[hd[f"y_d{D}"] == 1].copy()
    pos["score"] = pos.ons_obs - pos.onset_med
    cal = pos[pos.year <= 2023]; tst = pos[pos.year >= 2024]
    if len(cal) < 5:
        continue
    q_lo, q_hi = np.percentile(cal.score, [2.5, 97.5])
    for nm, g in [("cal", cal), ("test", tst)]:
        if len(g) == 0:
            continue
        cov = ((g.ons_obs >= g.onset_med + q_lo) & (g.ons_obs <= g.onset_med + q_hi)).mean()
        cov_rows.append(dict(horizon=D, set=nm, n=len(g), coverage=float(cov),
                             width=float(q_hi - q_lo), q_lo=float(q_lo), q_hi=float(q_hi)))
cov = pd.DataFrame(cov_rows)

# ------------------------------------------------------------------ event forecasts
ev_rows = []
for t0 in pd.date_range("2026-08-19", "2026-08-26", freq="D"):
    st = make_state(t0)
    sub = fd[(fd.date >= t0) & (fd.date <= t0 + pd.Timedelta(days=HMAX - 1))]
    H = len(sub)
    if H == 0:
        continue
    CTI = run_ensemble(t0, sub["Tmean_cal"].values, sub["VPD_qm"].values,
                       sub["P_cal"].values, st, N_ENS, RNG_E)
    s = onset_stats(CTI)
    dates = pd.to_datetime(sub["date"].values)
    eidx = int(np.where(dates == EVT)[0][0]) if (dates == EVT).any() else -1
    red = CTI >= CSTAR
    p_event = float(red[:, :eidx + 1].any(1).mean()) if eidx >= 0 else np.nan
    p_event_day = float(red[:, eidx].mean()) if eidx >= 0 else np.nan
    lead_h = (EVT_T - (t0 + pd.Timedelta(hours=8))).total_seconds() / 3600
    p_cal = float(iso[H].predict([s["p_any"]])[0]) if H in iso else np.nan
    ev_rows.append(dict(issue=t0, H=H, window_end=dates[-1].date(),
                        contains_event=int(eidx >= 0), event_index=eidx,
                        p_any_raw=s["p_any"], p_any_cal=p_cal,
                        p_event_raw=p_event, p_event_day_raw=p_event_day,
                        P24=s["p24"], P48=s["p48"], P72=s["p72"],
                        onset_med=t0 + pd.Timedelta(days=s["onset_med"]) if np.isfinite(s["onset_med"]) else pd.NaT,
                        onset_lo=t0 + pd.Timedelta(days=s["onset_lo"]) if np.isfinite(s["onset_lo"]) else pd.NaT,
                        onset_hi=t0 + pd.Timedelta(days=s["onset_hi"]) if np.isfinite(s["onset_hi"]) else pd.NaT,
                        lead_h=lead_h))
ev = pd.DataFrame(ev_rows)
print("\nH=7 event forecasts\n", ev.round(3).to_string(index=False))
ev.to_csv(os.path.join(OUT, "P1_A6_h7_event_forecast.csv"), index=False, float_format="%.4f")
hd.to_csv(os.path.join(OUT, "P1_A6_h7_hindcast_cases.csv"), index=False, float_format="%.4f")
lt.to_csv(os.path.join(OUT, "P1_A6_h7_leadtime.csv"), index=False, float_format="%.4f")
cov.to_csv(os.path.join(OUT, "P1_A6_h7_conformal.csv"), index=False, float_format="%.4f")
boot.to_csv(os.path.join(OUT, "P1_A6_h7_bootstrap.csv"), index=False, float_format="%.4f")

# ------------------------------------------------------------------ summary
with open(os.path.join(OUT, "P1_A6_h7_summary.md"), "w", encoding="utf-8") as f:
    f.write("# P1-A6/A9：H=7 长视野重分析与事件提前量\n\n")
    f.write("**生成**：2026-10-29  \n**脚本**：`work/analysis/p1_a_horizon7.py`  \n")
    f.write(f"**校准截止**：{CUTOFF.date()}（事件日不进入 GAM/PFAC/VPD_qm/SIG_*）  \n")
    f.write(f"**台站海拔**：{Z_ST:.0f} m（P0-9）  \n\n")
    f.write("## 1. 标定参数（事件排除）\n\n")
    f.write(f"- GAM = {GAM*1000:.3f} K/km；PFAC = {PFAC:.4f}；TM_BIAS = {TM_BIAS:.3f} °C\n")
    f.write(f"- SIG_T = {SIG_T:.3f} °C，RHO_T = {RHO_T:.3f}；SIG_VPD = {SIG_V:.3f} kPa；"
            f"SIG_Plog = {SIG_P:.2f}\n\n")
    f.write(f"## 2. H=7 perfect-prog 回测（n={len(hd)}，正例={int(hd.any_obs.sum())}）\n\n")
    f.write("| D | 正例 | 基率 | raw BSS | cal BSS | AUC | POD | FAR |\n")
    f.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for D in range(1, HMAX + 1):
        r = lt[(lt.horizon == D) & (lt.track == "calibrated")].iloc[0]
        rr = lt[(lt.horizon == D) & (lt.track == "raw")].iloc[0]
        f.write(f"| {D} | {int(r.pos)} | {r.base:.3f} | {rr.bss:.3f} | {r.bss:.3f} | "
                f"{r.auc:.3f} | {r.pod:.2f} | {r.far:.2f} |\n")
    f.write("\n### 2.1 Block bootstrap（10 天块，1000 次）\n\n")
    f.write("| D | 轨道 | BSS | 95% CI | AUC | 95% CI |\n|---:|---|---:|---|---:|---|\n")
    for _, r in boot.iterrows():
        f.write(f"| {int(r.horizon)} | {r.track} | {r.bss:.3f} | "
                f"[{r.bss_lo:.3f}, {r.bss_hi:.3f}] | {r.auc:.3f} | "
                f"[{r.auc_lo:.3f}, {r.auc_hi:.3f}] |\n")
    f.write("\n## 3. 共形覆盖（正例）\n\n")
    f.write(cov.to_markdown(index=False))
    f.write("\n\n## 4. 事件起报（H=7）\n\n")
    f.write(ev.round(3).to_markdown(index=False))
    f.write("\n\n**读法**：\n\n")
    f.write("- `p_event_raw` 是成员中“在 08-26 当天或之前首达 C*”的比例；"
            "只有窗口包含 08-26 的起报才有值。\n")
    f.write("- `p_any_raw` 是整个 H 天窗口内首达的比例（H=7 时 08-21 的窗口到 08-27，"
            "包含事件日之后一天）。\n")
    f.write("- `p_any_cal` 用同 horizon 的 isotonic 映射；08-20 的 H=7 窗口止于 08-26，"
            "故其 p_any 就是 event-specific 概率。\n")
    f.write("- 首个 `p_event_raw ≥ 0.05` 的起报即“首个看见事件日的预警”；"
            "首个 `p_any_cal ≥ 0.05` 的起报是“首个校准红警”。两者可能不同。\n")
print("\nDONE")
