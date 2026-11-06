# -*- coding: utf-8 -*-
"""Targeted GEFS check for the 26 Aug 2026 event.

The operational hindcast loop in 10_gefs_leadtime.py skips any issue whose
full 7-day verification window is not in daily.csv (which ends 2026-08-26).
Issue 2026-08-21 + 5 days = 2026-08-26, so the event day *is* an anchor lead
(D=5) but the loop drops it because t0+6 = 2026-08-27 is missing.  This script
runs that one issue without the completeness filter and reports the fraction
of the 31 members whose CTI reaches C* on the event day.

It is a diagnostic only; it does not modify the operational results.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
CODE = os.path.join(WORK, "code")
sys.path.insert(0, CODE)  # use work/code/ctilib.py (P0-7 elevation fix)

spec = importlib.util.spec_from_file_location(
    "gefs_leadtime", os.path.join(CODE, "10_gefs_leadtime.py"))
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)

# Sanity check: the reconstructed cti_from must reproduce the stored CTI series.
_o = G.CT.obs_cti(G.CT.load_obs())
_ev = _o[_o.date == pd.Timestamp("2026-08-26")].iloc[0]
_cti_re = float(_ev.CTI)
_ref = pd.read_csv(os.path.join(WORK, "results", "p2_cti_series.csv"),
                   parse_dates=["date"])
_cti_st = float(_ref.loc[_ref.date == pd.Timestamp("2026-08-26"), "CTI"].iloc[0])
print("ctilib CTI check:", round(_cti_re, 4), "vs stored", round(_cti_st, 4))

mem = G.load_members()
print("GEFS members:", mem.issue.nunique(), "issues,", mem.member.nunique(), "members")

m = G.pair_with_obs(mem)
tr = m[m.issue.dt.year <= G.TRAIN_END]
cal = G.calibrate(tr)
memc = G.apply_cal_members(mem, cal)

obs = G.CT.obs_cti(G.CT.load_obs())
t0 = pd.Timestamp("2026-08-21")
g0 = memc[memc.issue == t0]
members = sorted(g0.member.unique())
print("issue", t0.date(), "members", len(members), "rows", len(g0))

arr = G.member_arrays(g0, members)
st = G.make_state(obs, t0)
fcT, fcV, fcP = arr
CTI = G.run_members(fcT, fcV, fcP, st)
print("CTI shape", CTI.shape, "C*", G.CSTAR)
print("per-lead mean CTI", np.round(CTI.mean(0), 3).tolist())
print("per-lead exceedance", np.round((CTI >= G.CSTAR).mean(0), 3).tolist())
print("day5 Tmean mean/min/max %.2f/%.2f/%.2f" % (fcT[:, 4].mean(), fcT[:, 4].min(), fcT[:, 4].max()))
print("day5 VPD   mean/min/max %.2f/%.2f/%.2f" % (fcV[:, 4].mean(), fcV[:, 4].min(), fcV[:, 4].max()))
print("day5 P     mean/min/max %.2f/%.2f/%.2f" % (fcP[:, 4].mean(), fcP[:, 4].min(), fcP[:, 4].max()))
_o5 = obs[obs.date == pd.Timestamp("2026-08-26")].iloc[0]
print("observed 2026-08-26: Tmean %.2f VPD %.2f P %.2f" % (_o5.Ta_2_mean, _o5.VPD_kPa_max, _o5.Precip_sum))

# Event day is 2026-08-26 = t0 + 5 d = column index 4 (days 1..7).
k = 4
print("event-day (D=5) exceedance raw: %.4f" % float((CTI[:, k] >= G.CSTAR).mean()))
print("event-day (D=5) mean CTI: %.3f  max: %.3f" % (float(CTI[:, k].mean()), float(CTI[:, k].max())))

# Inflated members: same factors as the operational script.
sd = G.spread_diagnostic(mem[mem.issue.dt.year <= G.TRAIN_END], tr, cal).set_index("day")
f = G.inflation_factors(sd, cal)
fT, fV, fP = f["T"], f["V"], f["P"]
fTc, fVc, fPc = G.fac_col(fT), G.fac_col(fV), G.fac_col(fP)
CTIi = G.run_members(G.inflate(fcT, fTc), np.clip(G.inflate(fcV, fVc), 1e-3, None),
                     G.inflate(fcP, fPc), st)
print("per-lead exceedance inflated", np.round((CTIi >= G.CSTAR).mean(0), 3).tolist())
print("event-day (D=5) exceedance inflated: %.4f" % float((CTIi[:, k] >= G.CSTAR).mean()))

# Pipeline validation against a stored hindcast case (issue 2026-08-18).
# This checks load_members -> calibrate -> run_members -> onset_prob against
# the existing gefs_hindcast_cases.csv values.
t0v = pd.Timestamp("2026-08-18")
g0v = memc[memc.issue == t0v]
arrv = G.member_arrays(g0v, sorted(g0v.member.unique()))
stv = G.make_state(obs, t0v)
CTIv = G.run_members(*arrv, stv)
pv = G.onset_prob(CTIv, G.ANCH)
CTIiv = G.run_members(G.inflate(arrv[0], fTc), np.clip(G.inflate(arrv[1], fVc), 1e-3, None),
                      G.inflate(arrv[2], fPc), stv)
piv = G.onset_prob(CTIiv, G.ANCH)
stored = pd.read_csv(os.path.join(WORK, "results", "gefs_hindcast_cases.csv"))
srow = stored[pd.to_datetime(stored.issue) == t0v].iloc[0]
print("validation 08-18: p_d5 %.4f (stored %.4f), pinf_d5 %.4f (stored %.4f)"
      % (pv[5], srow.p_d5, piv[5], srow.pinf_d5))

# Observed value on the event day, for reference.
ev = pd.Timestamp("2026-08-26")
print("observed CTI 2026-08-26:",
      float(obs.loc[obs.date == ev, "CTI"].iloc[0]))
