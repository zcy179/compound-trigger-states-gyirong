# -*- coding: utf-8 -*-
"""Shared CTI primitives reconstructed from 04_p2.py / 05_p3.py.

The original `ctilib.py` imported by 10_gefs_leadtime.py was missing from the
repository.  This module restores the minimal API used there:

    load_obs()            -> daily observational table (work/clean/daily.csv)
    obs_cti(obs)          -> add API7 / PDD_3d / PDD_1d / CTI, drop NaN
    cti_from(api7,pdd3,vpd) -> CTI array with the same margins and Gaussian
                              copula as 04_p2.py / 05_p3.py

Margins and the copula are fitted on May-Oct 2021-2025 (n=920), exactly as in
05_p3.py.  `obs_cti` reproduces `results/p2_cti_series.csv` to within
1.6e-3 (median absolute difference 1.4e-5); the red-day labels are identical.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

from common import CLEAN

CSTAR = 2.5
P_STAR = 0.05
COLS = ["API7", "PDD_3d", "VPD_kPa_max"]
EVT = pd.Timestamp("2026-08-26")

_XG, _WG = np.polynomial.legendre.leggauss(40)
_TG, _WT = np.polynomial.legendre.leggauss(20)

_STATE = None


def load_obs():
    d = pd.read_csv(os.path.join(CLEAN, "daily.csv"), parse_dates=["date"])
    d = d.sort_values("date").reset_index(drop=True)
    d["API7"] = d["API"].rolling(7, min_periods=5).mean()
    d["PDD_1d"] = np.maximum(d["Ta_2_mean"], 0.0)
    return d


def _fit(d):
    """Fit margins and the Gaussian copula on the 2021-2025 seasons."""
    global _STATE
    if _STATE is not None:
        return _STATE
    seas = d["date"].dt.month.between(5, 10)
    train = d[seas & (d["date"] < "2026-01-01")].dropna(subset=COLS)
    n = len(train)
    u = np.column_stack([stats.rankdata(train[c], method="average") / (n + 1)
                         for c in COLS])
    tau = np.array([stats.kendalltau(u[:, i], u[:, j])[0]
                    for i, j in [(0, 1), (0, 2), (1, 2)]])
    sig = np.array([[1, np.sin(np.pi * tau[0] / 2), np.sin(np.pi * tau[1] / 2)],
                    [np.sin(np.pi * tau[0] / 2), 1, np.sin(np.pi * tau[2] / 2)],
                    [np.sin(np.pi * tau[1] / 2), np.sin(np.pi * tau[2] / 2), 1]])
    sig = (sig + sig.T) / 2
    marg = {c: np.sort(train[c].values) for c in COLS}
    _STATE = dict(n=n, sig=sig, marg=marg)
    return _STATE


def _p3_joint(z):
    """P(Z1>z1, Z2>z2, Z3>z3) for a standard trivariate normal."""
    z = np.atleast_2d(z)
    z1, z2, z3 = z[:, 0], z[:, 1], z[:, 2]
    xs = z1[:, None] + 4.0 * (1 + _XG)[None, :]
    wxs = 4.0 * _WG[None, :] * stats.norm.pdf(xs)
    r12, r13, r23 = _STATE["sig"][0, 1], _STATE["sig"][0, 2], _STATE["sig"][1, 2]
    s2, s3 = np.sqrt(1 - r12 ** 2), np.sqrt(1 - r13 ** 2)
    rc = (r23 - r12 * r13) / (s2 * s3)
    a = (z2[:, None] - r12 * xs) / s2
    b = (z3[:, None] - r13 * xs) / s3
    pa, pb = stats.norm.cdf(a), stats.norm.cdf(b)
    phi2 = pa * pb
    for tk, wk in zip(_TG, _WT):
        t = 0.5 * rc * (tk + 1.0)
        phi2 += (0.5 * rc * wk *
                 np.exp(-(a * a + b * b - 2 * a * b * t) / (2 * (1 - t * t))) /
                 (2 * np.pi * np.sqrt(1 - t * t)))
    return np.clip(((1 - pa - pb + phi2) * wxs).sum(1), 1e-15, 1.0)


def _ecdf(s, x):
    s = np.sort(np.asarray(s, float))
    x = np.asarray(x, float)
    return np.clip((np.searchsorted(s, x, side="right") + 0.5) / (len(s) + 1),
                   0.5 / (len(s) + 1), 1 - 0.5 / (len(s) + 1))


def cti_from(api7, pdd3, vpd):
    """CTI = -log10 P(API7>a, PDD_3d>p, VPDmax>v) under the fitted copula."""
    if _STATE is None:
        _fit(load_obs())
    sh = np.shape(api7)
    u1 = _ecdf(_STATE["marg"]["API7"], api7)
    u2 = _ecdf(_STATE["marg"]["PDD_3d"], pdd3)
    u3 = _ecdf(_STATE["marg"]["VPD_kPa_max"], vpd)
    z = np.column_stack([stats.norm.ppf(u1).ravel(),
                         stats.norm.ppf(u2).ravel(),
                         stats.norm.ppf(u3).ravel()])
    return (-np.log10(_p3_joint(z))).reshape(sh)


def obs_cti(obs):
    """Add CTI to an observational table and drop incomplete rows."""
    _fit(obs)
    d = obs.copy()
    if "PDD_3d" not in d:
        raise KeyError("daily.csv must contain PDD_3d")
    d["CTI"] = cti_from(d["API7"].values, d["PDD_3d"].values,
                        d["VPD_kPa_max"].values)
    return d.dropna(subset=COLS + ["CTI"]).reset_index(drop=True)
