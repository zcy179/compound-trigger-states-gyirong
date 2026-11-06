# -*- coding: utf-8 -*-
"""10 GEFS 真实集合回测 -> 集合概率的 lead-time 技能（与 08 的合成误差版对照）。

与 08 的关系（同一观测 CTI、同一 C*、同一评分，只换不确定性来源）：
  08：GFS 确定性预报 + 由 GFS-观测残差标定的 AR(1) 合成误差（1500 成员）；
  10：GEFS 31 个真实集合成员（09 取数），成员本身即扰动样本，
      只对集合平均做偏差订正（训练期 2021-2023），订正系数同套应用到每个成员，
      概率 = 成员中「未来 D 天内 CTI 首达 C*」的比例，不再叠加合成误差。

GEFS 取数约定（09）：起报日每 3 天一取（stride=3），lead 只存 24/72/120/168
（= 预报第 1/3/5/7 天），站点邻域 3x3 格点。因此：
  * 锚点日 = {1,3,5,7}；中间日 T/VPD 线性插值，P 只落在锚点日
    （降水乘性订正因子在训练期按同口径拟合，自动吸收采样稀疏的总量偏差）；
  * 评分只在锚点预见期 D=1/3/5/7 上做。

日量构造（每成员、每锚点日）：
  Tmean = (Tmax+Tmin)/2（GEFS 每 lead 只有 1 个瞬时 t2m，用日较差均值近似）
  VPDmax = es(Tmax) - RH(00z)·es(t2m_00z)（早晨 RH 露点代理，偏差由订正吸收）
  P = apcp（lead 24D 的 24h 累计，恰为第 D 天日窗）
地形订正与 07 完全一致：2x2 角点按环境递减率订正到站点海拔后 IDW。

输出 results/gefs_ens_daily.csv, gefs_calibration.csv,
        gefs_hindcast_cases.csv, gefs_leadtime.csv
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import RES
import ctilib as CT

CSTAR, P_STAR = CT.CSTAR, CT.P_STAR
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
GEFS = os.path.join(WORK, "data", "meteo", "gefs")
GFS = os.path.join(WORK, "data", "meteo", "gfs")

# 目标点：吉隆口岸（影响点）。GFS/GEFS 0.5° 网格在该处的高程是格元均值
# （GFS orog ≈4,103 m），因此按台站高程 4,140 m 做固定递减率订正；这不是
# 把预报降到口岸谷底（1,786 m），而是把网格温度映射到验证所用台站高度。
PORT = (28.28, 85.38)
Z_ST = 4140.0                                        # TPDC 台站高程（P0-7）
GAMMA = 6.5e-3
ANCH = [1, 3, 5, 7]                                   # 锚点预报日
MAXD = 7
TRAIN_END = 2023


def esat(t_c):
    return 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))


def corners(lats, lons, lat, lon):
    """包围 (lat,lon) 的 2x2 角点 + 反距离权重（与 07 同式）。"""
    i = int(np.searchsorted(lats, lat)) - 1
    j = int(np.searchsorted(lons, lon))
    i = min(max(i, 0), len(lats) - 2)
    j = min(max(j, 0), len(lons) - 2)
    pts = [(lats[i], lons[j]), (lats[i], lons[j + 1]),
           (lats[i + 1], lons[j]), (lats[i + 1], lons[j + 1])]
    w = []
    for la, lo in pts:
        dd = np.hypot(la - lat, (lo - lon) * np.cos(np.radians(lat))) * 111.0
        w.append(1.0 / max(dd, 1e-3) ** 2)
    w = np.array(w) / np.sum(w)
    return pts, w


# ---------------------------------------------------------------- A
def load_members():
    """全部 GEFS 文件 -> 站点订正后的逐成员日量表 (issue, day, member, ...)。"""
    orog = pd.read_csv(os.path.join(GFS, "orog_0p50.csv"))
    files = sorted(glob.glob(os.path.join(GEFS, "ens_*.csv")))
    if not files:
        raise SystemExit("no GEFS files yet")
    out = []
    for f in files:
        df = pd.read_csv(f)
        issue = pd.Timestamp(os.path.basename(f)[4:12])
        lats = np.sort(df.lat.unique())
        lons = np.sort(df.lon.unique())
        pts, w = corners(lats, lons, *PORT)
        z = np.array([orog[(np.isclose(orog.lat, p[0])) & (np.isclose(orog.lon, p[1]))]
                      .orog_m.iloc[0] for p in pts])
        dz = z - Z_ST
        cols = []
        for (pla, plo), wt, dzz in zip(pts, w, dz):
            s = df[(np.isclose(df.lat, pla)) & (np.isclose(df.lon, plo))].set_index(
                ["member", "lead"])
            cols.append(pd.DataFrame({
                "t2m": s.t2m - 273.15 + GAMMA * dzz,
                "tmax": s.tmax - 273.15 + GAMMA * dzz,
                "tmin": s.tmin - 273.15 + GAMMA * dzz,
                "rh": s.rh,
                "apcp": s.apcp}).mul(wt, axis=0))
        m = pd.concat(cols).groupby(level=["member", "lead"]).sum()
        m["day"] = m.index.get_level_values("lead") // 24
        m["Tmean"] = 0.5 * (m.tmax + m.tmin)
        m["VPD"] = (esat(m.tmax) - m.rh / 100.0 * esat(m.t2m)).clip(lower=0)
        g = m.reset_index()[["member", "day", "Tmean", "tmax", "tmin", "apcp", "VPD"]]
        g = g[g.day.isin(ANCH)].rename(columns={"tmax": "Tmax", "tmin": "Tmin",
                                                "apcp": "P", "VPD": "VPDmax"})
        g.insert(0, "issue", issue)
        out.append(g)
    d = pd.concat(out, ignore_index=True)
    d.to_csv(os.path.join(RES, "gefs_ens_daily.csv"), index=False, float_format="%.4f")
    return d


# ---------------------------------------------------------------- B
def pair_with_obs(mem):
    """集合平均与观测按 vdate=issue+(D-1) 对齐（与 08 align=0 同口径）。"""
    mean = mem.groupby(["issue", "day"], as_index=False)[
        ["Tmean", "Tmax", "Tmin", "P", "VPDmax"]].mean()
    obs = CT.obs_cti(CT.load_obs())
    st = obs[["date", "Ta_2_mean", "VPD_kPa_max", "Precip_sum"]].rename(columns={
        "Ta_2_mean": "o_Tmean", "VPD_kPa_max": "o_VPD", "Precip_sum": "o_P"})
    mean = mean.copy()
    mean["vdate"] = mean.issue + pd.to_timedelta(mean.day - 1, "D")
    m = mean.merge(st, left_on="vdate", right_on="date", how="inner")
    return m


def calibrate(m, mincal=30):
    """训练期 -> 锚点日的订正系数（与 08.calibrate 同式）+ 集合离散度诊断。"""
    pfac_all = float(m.o_P.sum() / max(m.P.sum(), 1e-6))
    rows = []
    for D in ANCH:
        s = m[m.day == D]
        if len(s) < mincal:
            continue
        bT = np.polyfit(s.Tmean, s.o_Tmean, 1)
        bV = np.polyfit(s.VPDmax, s.o_VPD, 1)
        rT = s.o_Tmean - np.polyval(bT, s.Tmean)
        rV = s.o_VPD - np.polyval(bV, s.VPDmax)
        rP = s.o_P - s.P * pfac_all
        rows.append(dict(day=D, n=len(s),
                         T_a=bT[1], T_b=bT[0], V_a=bV[1], V_b=bV[0], P_f=pfac_all,
                         rmseT=float(np.sqrt((rT ** 2).mean())),
                         rmseV=float(np.sqrt((rV ** 2).mean())),
                         rmseP=float(np.sqrt((rP ** 2).mean())),
                         r_det_T=float(np.corrcoef(s.Tmean, s.o_Tmean)[0, 1]),
                         r_det_V=float(np.corrcoef(s.VPDmax, s.o_VPD)[0, 1])))
    return pd.DataFrame(rows).set_index("day")


def spread_diagnostic(mem, m, cal):
    """集合离散度 vs 集合平均误差：欠散度诊断（论文可直接引用）。"""
    sd = mem.groupby(["issue", "day"], as_index=False)[
        ["Tmean", "VPDmax", "P"]].agg(["std", "mean"])
    rows = []
    for D in ANCH:
        s = sd.xs(D, level="day") if "day" in sd.index.names else sd[sd.day == D]
        e = m[m.day == D]
        errT = float(np.sqrt(np.mean((e.Tmean - e.o_Tmean) ** 2)))
        row = dict(day=D,
                   spreadT=float(s[("Tmean", "std")].mean()),
                   errT=errT,
                   spreadV=float(s[("VPDmax", "std")].mean()),
                   errV=float(np.sqrt(np.mean((e.VPDmax - e.o_VPD) ** 2))),
                   spreadP=float(s[("P", "std")].mean()),
                   errP=float(np.sqrt(np.mean((e.P - e.o_P) ** 2))))
        # Spread and error of the *calibrated* ensemble that is actually routed:
        # a linear calibration with slope s rescales the raw spread by s.
        for v, slope in (("T", cal.T_b[D]), ("V", cal.V_b[D]), ("P", cal.P_f[D])):
            row["spreadCal" + v] = row["spread" + v] * slope
            row["errCal" + v] = float(cal["rmse" + v][D])
        rows.append(row)
    return pd.DataFrame(rows)


def inflation_factors(sd, cal):
    """Variance inflation for calibrated member anomalies.

    f = (calibrated RMSE) / (calibrated ensemble spread).  Comparing the
    calibrated error with the *raw* spread would mis-scale VPD and
    precipitation, whose calibration slopes are far from one.
    """
    f = {}
    for v in ("T", "V", "P"):
        f[v] = {D: float(np.clip(sd.loc[D, "errCal" + v] /
                                 max(sd.loc[D, "spreadCal" + v], 1e-3), 1, 40))
                for D in ANCH}
    return f


# ---------------------------------------------------------------- C
def apply_cal_members(mem, cal):
    """把订正系数应用到每个成员（系数由集合平均拟合）。"""
    mem = mem.join(cal[["T_a", "T_b", "V_a", "V_b", "P_f"]], on="day")
    mem["Tc"] = np.polyval([mem.T_b, mem.T_a], mem.Tmean)
    mem["Vc"] = np.clip(np.polyval([mem.V_b, mem.V_a], mem.VPDmax), 1e-3, None)
    mem["Pc"] = mem.P * mem.P_f
    return mem


def member_arrays(g, members):
    """(member,day) 长表 -> 锚点宽表 -> 插值成 (nM, 7) 的 T/V/P 数组。"""
    w = g.pivot_table(index="member", columns="day", values=["Tc", "Vc", "Pc"])
    w = w.reindex(index=members, columns=pd.Index([1, 3, 5, 7], name="day"),
                  level=1)
    if w.isna().any().any():
        return None
    out = {}
    for v in ("Tc", "Vc", "Pc"):
        a = w[v].values.astype(float)                    # (nM, 4) 锚点
        full = np.empty((a.shape[0], MAXD))
        for k, D in enumerate(range(1, MAXD + 1)):
            if D in ANCH:
                full[:, k] = a[:, ANCH.index(D)]
            else:                                       # 中间日线性插值
                lo, hi = ANCH.index(D - 1), ANCH.index(D + 1)
                full[:, k] = 0.5 * (a[:, lo] + a[:, hi])
        if v == "Pc":                                    # P 只落锚点日
            full = np.zeros_like(full)
            for k, D in enumerate(ANCH):
                full[:, D - 1] = a[:, k]
        out[v] = full
    return out["Tc"], out["Vc"], out["Pc"]


def make_state(obs, t0):
    """与 05/08 同口径的起报状态（截至 t0-1）。"""
    i = obs.index[obs.date == t0][0]
    return dict(A=float(obs.loc[i - 1, "API"]),
                api_hist=obs.loc[i - 6:i - 1, "API"].values.astype(float),
                pdd_hist=obs.loc[i - 2:i - 1, "PDD_1d"].values.astype(float))


def run_members(fcT, fcV, fcP, state):
    """31 成员 -> (nM, 7) CTI。API 递推逐日 0.9 衰减，P 只在锚点日注入。"""
    n, H = fcT.shape
    A = np.zeros((n, H))
    api = np.full(n, state["A"])
    for k in range(H):
        api = 0.9 * api + fcP[:, k]
        A[:, k] = api
    B = np.concatenate([np.tile(state["api_hist"], (n, 1)), A], axis=1)
    api7 = np.column_stack([B[:, k:k + 7].mean(1) for k in range(H)])
    pdd = np.maximum(fcT, 0.0)
    ph = np.concatenate([np.tile(state["pdd_hist"], (n, 1)), pdd], axis=1)
    pdd3 = np.column_stack([ph[:, k:k + 3].sum(1) for k in range(H)])
    V = np.clip(fcV, 1e-3, None)
    return CT.cti_from(api7, pdd3, V)


def onset_prob(CTI, Ds):
    red = CTI >= CSTAR
    return {D: float(red[:, :D].any(1).mean()) for D in Ds}


def inflate(a, fcol):
    """等方差膨胀：把成员相对集合平均的异常按列（逐预报日）放大 fcol 倍。"""
    mu = a.mean(0, keepdims=True)
    return np.maximum(mu + (a - mu) * fcol[None, :], 0.0)


def fac_col(f_by_day):
    """锚点日因子 -> 7 列因子（中间日取相邻锚点均值）。"""
    col = np.empty(MAXD)
    for k, D in enumerate(range(1, MAXD + 1)):
        if D in ANCH:
            col[k] = f_by_day[D]
        else:
            col[k] = 0.5 * (f_by_day[D - 1] + f_by_day[D + 1])
    return col


# ---------------------------------------------------------------- D
def scores(p, y, clim):
    y = np.asarray(y, bool)
    p = np.asarray(p, float)
    br = float(np.mean((p - y) ** 2))
    cl = float(clim)
    bss = 1 - br / max(cl * (1 - cl), 1e-9)
    w = p >= P_STAR
    tp = int((w & y).sum()); fp = int((w & ~y).sum()); fn = int((~w & y).sum())
    auc = float(roc_auc_score(y.astype(int), p)) if 0 < y.sum() < len(y) else np.nan
    return dict(n=len(y), pos=int(y.sum()), base=cl, brier=br, bss=bss,
                pod=tp / max(tp + fn, 1), far=fp / max(tp + fp, 1),
                csi=tp / max(tp + fp + fn, 1), nwarn=int(w.sum()), auc=auc)


def main():
    mem = load_members()
    print("GEFS members: issues=%d members=%d rows=%d"
          % (mem.issue.nunique(), mem.member.nunique(), len(mem)), flush=True)

    m = pair_with_obs(mem)
    tr = m[m.issue.dt.year <= TRAIN_END]
    print("\n=== 标定（训练期 %s, n=%d）===" % (tr.issue.min().date(), len(tr)))
    cal = calibrate(tr)
    print(cal[["n", "T_a", "T_b", "V_a", "V_b", "P_f", "rmseT", "r_det_T"]]
          .round(3).to_string())
    cal.to_csv(os.path.join(RES, "gefs_calibration.csv"), float_format="%.4f")

    print("\n=== 离散度诊断（训练期：成员 std vs 订正后集合平均误差）===")
    sd = spread_diagnostic(mem[mem.issue.dt.year <= TRAIN_END], tr, cal)
    sd = sd.set_index("day")
    print(sd.round(3).to_string())
    f = inflation_factors(sd, cal)
    fT, fV, fP = f["T"], f["V"], f["P"]
    print("\n膨胀因子 f=误差/离散度: T %s  VPD %s  P %s"
          % ({D: round(fT[D], 1) for D in ANCH},
             {D: round(fV[D], 1) for D in ANCH},
             {D: round(fP[D], 1) for D in ANCH}))
    sd_out = sd.copy()
    sd_out["fT"] = [fT[D] for D in sd_out.index]
    sd_out["fV"] = [fV[D] for D in sd_out.index]
    sd_out["fP"] = [fP[D] for D in sd_out.index]
    sd_out.to_csv(os.path.join(RES, "gefs_spread_error.csv"),
                  float_format="%.4f")
    fTc, fVc, fPc = fac_col(fT), fac_col(fV), fac_col(fP)

    memc = apply_cal_members(mem, cal)
    obs = CT.obs_cti(CT.load_obs())
    oidx = obs.set_index("date")

    rows = []
    for t0, g0 in memc.groupby("issue"):
        if t0.year <= TRAIN_END or t0 not in oidx.index:
            continue
        if t0 - pd.Timedelta(days=1) not in oidx.index:
            continue
        if oidx.loc[t0 - pd.Timedelta(days=1), "CTI"] >= CSTAR:
            continue
        if t0 + pd.Timedelta(days=MAXD - 1) not in oidx.index:
            continue
        members = sorted(g0.member.unique())
        if len(members) < 20:
            continue
        try:
            arr = member_arrays(g0, members)
            st = make_state(obs, t0)
        except (IndexError, KeyError):
            continue
        if arr is None:
            continue
        fcT, fcV, fcP = arr
        CTI = run_members(fcT, fcV, fcP, st)
        p = onset_prob(CTI, ANCH)
        CTIi = run_members(inflate(fcT, fTc), np.clip(inflate(fcV, fVc), 1e-3, None),
                           inflate(fcP, fPc), st)
        pi = onset_prob(CTIi, ANCH)
        fut = oidx.loc[[t0 + pd.Timedelta(days=d - 1) for d in range(1, MAXD + 1)],
                       "CTI"].values
        red = fut >= CSTAR
        rows.append(dict(issue=t0, year=t0.year, nmem=len(members),
                         **{"p_d%d" % D: p[D] for D in ANCH},
                         **{"pinf_d%d" % D: pi[D] for D in ANCH},
                         **{"y_d%d" % D: int(red[:D].any()) for D in ANCH},
                         cti_obs1=float(fut[0]),
                         cti_med1=float(np.median(CTI[:, 0])),
                         cti_q10=float(np.quantile(CTI[:, 0], 0.1)),
                         cti_q90=float(np.quantile(CTI[:, 0], 0.9))))
    hd = pd.DataFrame(rows)
    hd.to_csv(os.path.join(RES, "gefs_hindcast_cases.csv"), index=False,
              float_format="%.4f")
    print("\n=== 检验期起报数 %d（%s ~ %s）==="
          % (len(hd), hd.issue.min().date(), hd.issue.max().date()))

    out = []
    for D in ANCH:
        for var, col in (("raw", "p_d%d" % D), ("inflated", "pinf_d%d" % D)):
            sc = scores(hd[col], hd["y_d%d" % D], hd["y_d%d" % D].mean())
            sc["lead"] = D
            sc["variant"] = var
            out.append(sc)
    lt = pd.DataFrame(out)[["variant", "lead", "n", "pos", "base", "brier", "bss",
                            "pod", "far", "csi", "nwarn", "auc"]]
    print("\n=== lead-time 技能（GEFS 真实集合：原始 vs 膨胀订正）===")
    print(lt.round(3).to_string(index=False))
    lt.to_csv(os.path.join(RES, "gefs_leadtime.csv"), index=False,
              float_format="%.4f")

    gp = os.path.join(RES, "gfs_leadtime.csv")
    if os.path.exists(gp):
        g = pd.read_csv(gp)
        g = g[g.lead.isin(ANCH)]
        cmp = lt[lt.variant == "inflated"].merge(g, on="lead", suffixes=("_gefs", "_gfs"))
        print("\n=== 对照（GEFS 膨胀集合 vs GFS+合成误差，同检验期不同起报集合）===")
        print(cmp[["lead", "bss_gefs", "bss_gfs", "csi_gefs", "csi_gfs",
                   "pod_gefs", "pod_gfs", "auc_gefs", "auc_gfs"]]
              .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
