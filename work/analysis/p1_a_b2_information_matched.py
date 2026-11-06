# -*- coding: utf-8 -*-
"""B2 (review major point 6): information-set-matched comparators.

Question
--------
Table 11 compares the copula/first-passage engine with three predictors that
see only quantities available at the issue date, while the engine additionally
sees the meteorology of the D forecast days.  The BSS margin (+0.11 at D=1 to
+0.63 at D=7) therefore confounds "the method" with "the extra information".
The review asks for an information-matched comparator: GFS-predicted drivers
passed through the same threshold, or a logistic model fitted on GFS-predicted
drivers.

What this script actually computes
----------------------------------
Two new logistic arms, on the identical 976 issues / identical y_dD labels /
identical train-test split as P5_id_baseline.py and P6_baseline_table.py:

  * future_observed : three drivers taken from the D-day FUTURE OBSERVED window.
                      This is the exact information set of the perfect-prog
                      engine (p1_a_horizon7.py `run_ensemble` receives
                      `sub["Ta_2_mean"]`, `sub["VPD_kPa_max"]`,
                      `sub["Precip_sum"]` from the observed record).  The
                      engine - this_arm difference isolates the METHOD.
  * gfs_predicted   : the same three drivers reconstructed from the D-day GFS
                      deterministic forecast archive (gfs_daily.csv, NOAA GFS
                      0.5 deg, 7 leads per issue).  This is the review's
                      requested arm.  The future_observed - gfs_predicted
                      difference isolates the cost of real forecast error.

Each arm uses three features (window maximum of API7, PDD_3d and VPDmax), the
same three variables the CTI is built from, but NO copula, NO ensemble and NO
first-passage calculation.  The existing logistic (issue-date drivers only) and
the engine are taken verbatim from P6_baseline_comparison.csv.

Hard assertions (the script aborts if any fails)
------------------------------------------------
  1. y_dD labels are reproducible cell-by-cell from the archived CTI series.
  2. Re-running the existing issue-date logistic reproduces P6 to <1e-4.
  3. All 976 issues have finite features in both new arms.
  4. All 976 issues are covered by the GFS archive.

Outputs
-------
  work/results/paperA_b2_information_matched.csv
  work/analysis/B2_information_matched_verification.md
"""
import os
import io
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
W = lambda *p: os.path.join(ROOT, *p)
RES = W("work", "results")
ANA = W("work", "analysis")
CSTAR = 2.5
FEATS = ["API7", "PDD_3d", "VPD_kPa_max"]
SRC_MAIN = "port"          # grid point closest in elevation to the 4140 m station
SRCS = ["port", "county", "source"]


def f3(x, sign=False):
    """Round half up to three decimals and use the manuscript's U+2212 minus."""
    q = Decimal(str(round(float(x), 6))).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP)
    s = ("%+.3f" % q) if sign else ("%.3f" % q)
    return s.replace("-", "\u2212")


# ---------------------------------------------------------------- load
h = pd.read_csv(W("work", "analysis", "P1_A6_h7_hindcast_cases.csv"),
                parse_dates=["issue"])
cti = pd.read_csv(os.path.join(RES, "p2_cti_series.csv"), parse_dates=["date"])
daily = pd.read_csv(W("work", "clean", "daily.csv"),
                    parse_dates=["date"]).sort_values("date").reset_index(drop=True)
p6 = pd.read_csv(os.path.join(ANA, "P6_baseline_comparison.csv"),
                 encoding="utf-8-sig")
gfs = pd.read_csv(os.path.join(RES, "gfs_daily.csv"))
gfs["issue"] = pd.to_datetime(gfs.issue)

# ---------------------------------------------------------------- assertion 1
cser = cti.set_index("date")["CTI"]
for D in range(1, 8):
    recon = []
    for t0 in h.issue:
        v = cser.reindex(pd.date_range(t0, periods=D, freq="D")).values
        recon.append(int(bool((v >= CSTAR).any())) if np.isfinite(v).all() else -1)
    mism = int((np.asarray(recon) != h["y_d%d" % D].values).sum())
    assert mism == 0, "A1: y_d%d mismatches %d cells" % (D, mism)
print("A1 OK  y_d1..y_d7 reproduce the archived CTI series on all 976 issues")

# ---------------------------------------------------------------- baseline X
X_issue = cti.set_index("date")[FEATS].reindex(h.issue).reset_index(drop=True)
# cross-check against P5's own construction (clean/daily rolling API7)
_d = daily.copy()
_d["API7"] = _d["API"].rolling(7, min_periods=5).mean()
X_p5 = _d.set_index("date")[FEATS].reindex(h.issue).reset_index(drop=True)
for c in FEATS:
    a, b = X_issue[c].values, X_p5[c].values
    ok = np.isfinite(a) & np.isfinite(b)
    assert np.allclose(a[ok], b[ok], atol=1e-5), "A1b: feature %s differs" % c


def scores(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    clim = np.full(len(y), y.mean())
    bs = float(np.mean((p - y) ** 2))
    br = float(np.mean((clim - y) ** 2))
    bss = 1.0 - bs / br if br > 0 else np.nan
    auc = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan
    return bss, auc


def fit_arm(X, tag, calibrate=False, aggregate="max"):
    """One logistic arm, fitted per horizon on <=2023, tested on >=2024.

    X may be a [n, 3] DataFrame (issue-date drivers, no window) or a
    [n, 7, 3] ndarray (per-day drivers).  In the latter case the features are
    reduced over the FIRST D days only, so each horizon sees exactly the same
    window as its cumulative label y_dD.  Reducing over all seven days at every
    horizon would leak days t0+D..t0+6 into the short-horizon arms.
    """
    is_cube = isinstance(X, np.ndarray) and X.ndim == 3
    out = []
    for D in range(1, 8):
        y_all = h["y_d%d" % D].values.astype(float)
        if is_cube:
            w = X[:, :D, :]
            Xd = (np.nanmax(w, axis=1) if aggregate == "max"
                  else np.nanmean(w, axis=1))
            Xd = pd.DataFrame(Xd, columns=FEATS)
        else:
            Xd = X
        ok = np.isfinite(Xd).all(1).values
        tr = (h.year <= 2023).values & ok
        te = (h.year >= 2024).values & ok
        sc = StandardScaler().fit(Xd[tr])
        clf = LogisticRegression(max_iter=2000).fit(
            sc.transform(Xd[tr]), y_all[tr])
        p = clf.predict_proba(sc.transform(Xd[te]))[:, 1]
        bss, auc = scores(y_all[te], p)
        bss_c = np.nan
        if calibrate:
            iso = IsotonicRegression(out_of_bounds="clip")
            p_tr = clf.predict_proba(sc.transform(Xd[tr]))[:, 1]
            iso.fit(p_tr, y_all[tr])
            bss_c, _ = scores(y_all[te], iso.predict(p))
        out.append(dict(arm=tag, D=D, n_train=int(tr.sum()), n_test=int(te.sum()),
                        base_rate=float(y_all[te].mean()),
                        positives=int(y_all[te].sum()),
                        bss=bss, auc=auc, bss_calibrated=bss_c,
                        aggregate=aggregate))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- assertion 2
rep = fit_arm(X_issue, "logistic_issue_day")
chk = p6[["D", "logit_bss", "logit_auc"]].merge(rep, on="D")
assert float((chk.bss - chk.logit_bss).abs().max()) < 1e-4, "A2: BSS mismatch"
assert float((chk.auc - chk.logit_auc).abs().max()) < 1e-4, "A2: AUC mismatch"
print("A2 OK  issue-date logistic reproduces Table 11 (max |dBSS|=%.2g, |dAUC|=%.2g)"
      % (float((chk.bss - chk.logit_bss).abs().max()),
         float((chk.auc - chk.logit_auc).abs().max())))

# ------------------------------------------------- future OBSERVED window features
s = cti.set_index("date")[FEATS]


def window_features(source):
    """[976, 7, 3] array: for each issue and each day k, the three drivers."""
    arr = np.full((len(h), 7, 3), np.nan)
    for i, t0 in enumerate(h.issue):
        arr[i] = source(t0)
    return arr


def obs_source(t0):
    return s.reindex(pd.date_range(t0, periods=7, freq="D"))[FEATS].values.astype(float)


CUBE_OBS = window_features(obs_source)
assert np.isfinite(CUBE_OBS).all(), "A3: future-observed features have gaps"
fut = fit_arm(CUBE_OBS, "logistic_future_observed", calibrate=True)
fut_mean = fit_arm(CUBE_OBS, "logistic_future_observed_mean", aggregate="mean")

# ------------------------------------------------- GFS predicted window features
api_series = daily.set_index("date")["API"]
tmean_series = daily.set_index("date")["Ta_2_mean"]
gfs_by_src = {k: gfs[gfs.src == k].set_index(["issue", "day"]).sort_index()
              for k in SRCS}


def gfs_source(src):
    tab = gfs_by_src[src]

    def _one(t0):
        try:
            sub = tab.loc[t0].sort_index()
        except KeyError:
            return np.full((7, 3), np.nan)
        if len(sub) != 7:
            return np.full((7, 3), np.nan)
        P = sub["P"].values.astype(float)
        T = sub["Tmean"].values.astype(float)
        V = sub["VPDmax"].values.astype(float)
        hist_api = api_series.reindex(
            pd.date_range(t0 - pd.Timedelta(days=6), periods=6, freq="D")).values.astype(float)
        hist_t = tmean_series.reindex(
            pd.date_range(t0 - pd.Timedelta(days=2), periods=2, freq="D")).values.astype(float)
        if not (np.isfinite(hist_api).all() and np.isfinite(hist_t).all()):
            return np.full((7, 3), np.nan)
        # identical recursion to p1_a_horizon7.py:run_ensemble, without noise
        api = float(api_series.reindex(
            pd.date_range(t0 - pd.Timedelta(days=1), periods=1, freq="D")).values[0])
        A = []
        for p in P:
            api = 0.9 * api + p
            A.append(api)
        B = np.concatenate([hist_api, np.asarray(A)])
        api7 = np.array([B[k:k + 7].mean() for k in range(7)])
        pdd_hist = np.maximum(hist_t, 0.0)
        pdd_day = np.maximum(T, 0.0)
        ph = np.concatenate([pdd_hist, pdd_day])
        pdd3 = np.array([ph[k:k + 3].sum() for k in range(7)])
        return np.column_stack([api7, pdd3, V])
    return _one


covered = sum(1 for t0 in h.issue if t0 in set(gfs.issue))
assert covered == len(h), "A4: GFS archive covers %d/%d issues" % (covered, len(h))
print("A4 OK  GFS archive covers all %d issues" % covered)

arms = {}
for src in SRCS:
    cube = window_features(gfs_source(src))
    assert np.isfinite(cube).all(), "A3: GFS(%s) features have gaps" % src
    arms[src] = fit_arm(cube, "logistic_gfs_" + src,
                        calibrate=(src == SRC_MAIN))
print("A3 OK  all 976 issues have finite features in both new arms")

# ---------------------------------------------------------------- assemble
engine = p6[["D", "engine_bss", "engine_auc"]].rename(
    columns={"engine_bss": "bss", "engine_auc": "auc"})
engine.insert(0, "arm", "engine_perfect_prog_calibrated")
engine["n_test"] = p6["n"].values
engine["base_rate"] = p6["base_rate"].values
engine["bss_calibrated"] = np.nan

all_rows = pd.concat(
    [engine, rep, fut, arms[SRC_MAIN]] + [arms[k] for k in SRCS if k != SRC_MAIN]
    + [fut_mean], ignore_index=True, sort=False)
all_rows.to_csv(os.path.join(RES, "paperA_b2_information_matched.csv"),
                index=False, float_format="%.6f")

# ---------------------------------------------------------------- table 13
main = {"engine": engine.set_index("D"),
        "issue": rep.set_index("D"),
        "fut": fut.set_index("D"),
        "gfs": arms[SRC_MAIN].set_index("D")}


def panel(metric, sign=False):
    lines = ["| D | Base rate | Engine (perfect-prog) | Logistic, issue-day | "
             "Logistic, future observed | Logistic, GFS-predicted |",
             "|---:|---:|---:|---:|---:|---:|"]
    for D in range(1, 8):
        vals = [f3(main[k].loc[D, metric], sign) for k in
                ("engine", "issue", "fut", "gfs")]
        lines.append("| %d | %s | %s | %s | %s | %s |" %
                     (D, f3(main["fut"].loc[D, "base_rate"]), *vals))
    return "\n".join(lines)


d7 = {k: main[k].loc[7] for k in main}

# ------------------------------------------------- assertions 5-7: internal logic
r1_f, r1_i = fut.set_index("D").loc[1], rep.set_index("D").loc[1]
assert abs(r1_f.bss - r1_i.bss) < 1e-8 and abs(r1_f.auc - r1_i.auc) < 1e-8, (
    "A5: at D=1 the future-observed window IS the issue date, so the two arms "
    "must coincide; a mismatch means the per-horizon window slicing is wrong")
print("A5 OK  D=1 future-observed arm equals the issue-date arm (window check)")

decomp = ((d7["engine"].bss - d7["fut"].bss) +
          (d7["fut"].bss - d7["gfs"].bss) +
          (d7["gfs"].bss - d7["issue"].bss))
assert abs(decomp - (d7["engine"].bss - d7["issue"].bss)) < 1e-12, (
    "A6: three-term decomposition identity")
assert all(arms[s].set_index("D").loc[7, "bss"] < d7["fut"].bss for s in SRCS), (
    "A7: a GFS grid point outperforms the noise-free future-observed arm")
print("A6/A7 OK  decomposition exact; all three GFS points below the PP arm")

print("\n=== Table 13 (draft) ===")
print("Panel A — Brier skill score")
print(panel("bss", True))
print("\nPanel B — ROC AUC")
print(panel("auc"))
print("\n=== D=7 summary ===")
for k, v in d7.items():
    print("  %-12s BSS=%+.4f AUC=%.4f" % (k, v.bss, v.auc))
print("  method increment (engine - future observed) BSS = %+.4f"
      % (d7["engine"].bss - d7["fut"].bss))
print("  forecast-error cost (future observed - GFS) BSS = %+.4f"
      % (d7["fut"].bss - d7["gfs"].bss))
print("  info gain (GFS - issue-day) BSS = %+.4f"
      % (d7["gfs"].bss - d7["issue"].bss))

# ---------------------------------------------------------------- method note
rob = []
for src in SRCS:
    r = arms[src].set_index("D").loc[7]
    rob.append("| %s | %s | %s |" % (src, f3(r.bss, True), f3(r.auc)))
r7 = fut_mean.set_index("D").loc[7]
rc = fut.set_index("D").loc[7]
rg = arms[SRC_MAIN].set_index("D").loc[7]

md = []
md.append("# B2 — information-set-matched comparators (2026-11-06)\n")
md.append("Review major point 6 asked for a comparator that sees the same future "
          "meteorology as the engine. This note documents the two arms added to "
          "Section 4.11 and the numeric output in "
          "`work/results/paperA_b2_information_matched.csv`. "
          "Script: **`work/analysis/p1_a_b2_information_matched.py`**.\n")
md.append("## Design\n")
md.append("- Sample: the identical 976 archived issues of `P1_A6_h7_hindcast_cases.csv` "
          "used by Table 11; train = 2021–2023 (539), test = 2024–2026 (437). "
          "Labels `y_dD` are the cumulative first-crossing labels of Section 3.3; "
          "they reproduce the archived CTI series on 976/976 issues at every D "
          "(assertion A1), so no new predictand is introduced.\n")
md.append("- Every arm is `StandardScaler` + `LogisticRegression(max_iter=2000)` "
          "fitted separately at each horizon, exactly as in `P5_id_baseline.py`. "
          "Brier skill uses the test-sample climatology as reference; AUC is "
          "calibration-invariant. Re-running the existing issue-date arm reproduces "
          "Table 11 to <1e-4 (assertion A2).\n")
md.append("- **future observed**: the three CTI drivers (`API7`, `PDD_3d`, "
          "`VPD_kPa_max`) reduced over the D-day *observed* future window by their "
          "maximum. `p1_a_horizon7.py:run_ensemble` is driven by exactly this "
          "window (`sub[...]` from the observed record), so this arm holds the "
          "engine's information set with no copula, no ensemble and no "
          "first-passage calculation. The engine–arm gap isolates the **method**.\n")
md.append("- **GFS predicted**: the same three drivers reconstructed from the "
          "archived NOAA GFS 0.5° deterministic forecasts (`gfs_daily.csv`, seven "
          "leads per issue, 1,100 issues, 100% coverage of the 976). The API and "
          "PDD recursions, the six-day API / two-day PDD history window and the "
          "0.9 decay are copied verbatim from `run_ensemble`, with the stochastic "
          "perturbations removed. The main result uses the `port` grid point "
          "(elevation 4103 m, closest to the 4140 m station and the same point as "
          "the event-track correction); all three grid points are reported below. "
          "This is the arm the review asked for.\n")
md.append("- The engine column is the isotonically calibrated perfect-prog value "
          "already in Table 11; the three logistic columns are raw probabilities. "
          "This is the pre-existing comparison convention of Table 11 (see "
          "`P6_baseline_comparison.md`). A calibrated variant of each new arm is "
          "reported under Robustness so the conclusion can be checked against it.\n")
md.append("- The review offered two variants: GFS drivers passed through the CTI "
          "threshold, or a logistic model fitted on GFS drivers. We implement the "
          "**logistic** variant. A deterministic 0/1 threshold forecast is "
          "structurally penalised by the Brier score on rare events (already noted "
          "in `P6_baseline_comparison.md`, where TSS is used instead), and "
          "recomputing a GFS-driven CTI would require applying station-fitted "
          "marginals to uncorrected 0.5° grid values, which would add a "
          "distribution-shift confound. A probabilistic arm is like-for-like with "
          "the engine on both BSS and AUC.\n")
md.append("## Table 13 — information-matched comparison, 437 held-out issues\n")
md.append("Panel A — Brier skill score (positive is better than climatology)\n")
md.append(panel("bss", True) + "\n")
md.append("Panel B — ROC AUC\n")
md.append(panel("auc") + "\n")
md.append("Reading the columns: `GFS − issue-day` is the value of future "
          "meteorological information under a real forecast; `future observed − "
          "GFS` is the cost of forecast error; **`engine − future observed` is the "
          "method increment that Table 11 could not identify**. All four columns "
          "use the same labels and the same test issues.\n")
md.append("## D = 7 decomposition\n")
md.append("| Quantity | BSS |\n|---|---:|")
md.append("| Engine − logistic (issue-day), the old +0.63 gap | %s |"
          % f3(d7["engine"].bss - d7["issue"].bss, True))
md.append("| Information: logistic (GFS) − logistic (issue-day) | %s |"
          % f3(d7["gfs"].bss - d7["issue"].bss, True))
md.append("| Forecast error: logistic (future observed) − logistic (GFS) | %s |"
          % f3(d7["fut"].bss - d7["gfs"].bss, True))
md.append("| **Method: engine − logistic (future observed)** | **%s** |"
          % f3(d7["engine"].bss - d7["fut"].bss, True))
md.append("\nThe three middle rows sum exactly to the first (assertion A6), so the "
          "old +0.63 gap is fully accounted for: roughly one third is future "
          "meteorological information under a real forecast, half is the cost of "
          "forecast error, and only the residual is attributable to the "
          "copula-and-first-passage construction. **The %s should be read as a "
          "conservative lower bound on the method increment**, because the "
          "future-observed arm is deliberately given the most favourable "
          "feature reduction (window maximum, no noise, same three drivers as "
          "the index). Weakening that arm raises the residual (see Robustness).\n"
          % f3(d7["engine"].bss - d7["fut"].bss, True))
md.append("## Robustness\n")
md.append("**Grid-point choice (GFS arm, D = 7).**\n")
md.append("| GFS grid point | BSS | AUC |\n|---|---:|---:|")
md.extend(rob)
md.append("\n**Feature aggregation — the method increment is NOT a point estimate.** "
          "Replacing the window maximum by the window mean in the future-observed "
          "arm gives D = 7 BSS %s and AUC %s (against %s / %s with the maximum), "
          "which raises the method increment from %s to **%s**. The increment is "
          "positive under both reductions, but its size depends strongly on how "
          "much structure the comparator is given. Reporting a single number "
          "would overstate the precision; the paper therefore quotes the "
          "strongest-comparator value (%s) as a lower bound and states the "
          "sensitivity.\n"
          % (f3(r7.bss, True), f3(r7.auc), f3(rc.bss, True), f3(rc.auc),
             f3(d7["engine"].bss - rc.bss, True),
             f3(d7["engine"].bss - r7.bss, True),
             f3(d7["engine"].bss - rc.bss, True)))
md.append("**Calibration.** After an isotonic map fitted on 2021–2023 (the same "
          "procedure as the engine), D = 7 BSS is %s for the future-observed arm "
          "and %s for the GFS arm (raw: %s / %s); the corresponding method "
          "increment is %s rather than %s. Calibration slightly *reduces* the "
          "future-observed arm's skill here, so the raw comparison is the more "
          "conservative one. AUC is invariant to the monotone map.\n"
          % (f3(rc.bss_calibrated, True), f3(rg.bss_calibrated, True),
             f3(rc.bss, True), f3(rg.bss, True),
             f3(d7["engine"].bss - rc.bss_calibrated, True),
             f3(d7["engine"].bss - rc.bss, True)))
md.append("## Limits (must be stated in the paper)\n")
md.append("1. This is still a comparison *inside* the self-referential framework: "
          "labels are generated by the CTI, and the new arms use the same three "
          "drivers. It separates the method from the information set, **not** the "
          "index from an external landslide record.\n")
md.append("2. The future-observed arm is a perfect-prog upper bound — it sees "
          "noise-free future meteorology. The GFS arm is the operationally "
          "reachable one, but it is compared with a perfect-prog engine, so its "
          "deficit also contains the engine's information advantage.\n")
md.append("3. GFS grid-point values are raw 0.5° forecasts with no lapse-rate or "
          "quantile correction; they are standardised and fitted within the GFS "
          "distribution, so each arm is internally consistent, but the GFS arm is "
          "not a like-for-like substitute for the station-calibrated drivers.\n")
md.append("4. Seven separate logistic fits on 539 training issues with 9–52 test "
          "positives; no intervals are attached. These are diagnostic comparisons, "
          "not a new validation sample.\n")

with io.open(os.path.join(ANA, "B2_information_matched_verification.md"),
             "w", encoding="utf-8") as fh:
    fh.write("\n".join(md) + "\n")

print("\nwrote work/results/paperA_b2_information_matched.csv (%d rows)" % len(all_rows))
print("wrote work/analysis/B2_information_matched_verification.md")
