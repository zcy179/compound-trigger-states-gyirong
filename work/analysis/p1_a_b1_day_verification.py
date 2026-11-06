# -*- coding: utf-8 -*-
"""B1: day-specific (first-passage-day) verification of the CTI ensemble.

Purpose
-------
The archived per-issue verification scores a CUMULATIVE predictand
    y_dk = 1{CTI >= C* at least once in the k days after issue},
with the matching ensemble frequency
    p_dk = P(first crossing within k days).
Because y_dk is monotone in k, its base rate grows with nominal lead
(GFS: 0.022 -> 0.136), and the cumulative BSS rises (0.176 -> 0.437).
A reviewer asked for a day-specific table in which the lead decay becomes
visible.

This script performs an exact ALGEBRAIC DECOMPOSITION of the archived
cumulative probabilities and labels, with no re-fitting and no new data:

    q_k = p_dk - p_d(k-1)     (p_d0 = 0)  -> P(first crossing ON day k)
    h_k = y_dk - y_d(k-1)     (y_d0 = 0)  -> 1{first crossing ON day k}

For the perfect-prog H=7 track h_k is checked against the archived
`ons_obs` column.  Scoring q_k against h_k at each k separately gives a
discrete-time hazard (first-passage-day) verification in which the base
rate no longer accumulates with k.

Why raw frequencies only
------------------------
The calibrated columns `pc_dk` come from ONE isotonic map per horizon,
fitted independently, so they are not guaranteed monotone across k
(166 negative differences, minimum -0.600 in the archived H=7 cases).
Differencing them would produce invalid probabilities.  The raw `p_dk`
are cumulative distribution functions by construction and their
differences are non-negative (0 negative cells, asserted below).
AUC is invariant to the within-horizon monotone isotonic maps anyway.

Inputs (all archived, read-only)
--------------------------------
  work/results/gfs_hindcast_cases.csv               455 GFS issues (raw)
  work/results/gfs_leadtime.csv                     reference metrics
  work/analysis/P1_A6_h7_hindcast_cases.csv        976 perfect-prog issues
  work/analysis/P1_A6_h7_leadtime.csv               reference metrics

Outputs
-------
  work/results/paperA_b1_day_verification.csv       full numeric table
  work/analysis/B1_day_specific_verification.md    method + ready-to-paste table
"""
import os
import io
import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
RES = os.path.join(WORK, "results")
OUT_CSV = os.path.join(RES, "paperA_b1_day_verification.csv")
OUT_MD = os.path.join(HERE, "B1_day_specific_verification.md")
KMAX = 7
TOL = 1e-4                      # archived CSVs are rounded to 4 decimals


def score(y, p):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    base = float(y.mean())
    if not (0 < y.sum() < len(y)) or not (0 < base < 1):
        return dict(base=base, brier=np.nan, bss=np.nan, auc=np.nan)
    brier = float(np.mean((p - y) ** 2))
    return dict(base=base, brier=brier,
                bss=1 - brier / (base * (1 - base)),
                auc=float(roc_auc_score(y, p)))


def load_track(df, label, pcol, holdout_mask=None, ref=None):
    """Return the day-specific table for one track plus a reproducibility check."""
    n_all = len(df)
    F = df[[pcol + str(k) for k in range(1, KMAX + 1)]].values.astype(float)
    Y = df[["y_d%d" % k for k in range(1, KMAX + 1)]].values.astype(int)
    Q = np.diff(np.column_stack([np.zeros(n_all), F]), axis=1)
    H = np.diff(np.column_stack([np.zeros(n_all, int), Y]), axis=1)

    # --- hard assertions: the decomposition must be a valid CDF difference ---
    assert Q.min() >= -1e-12, "%s: negative q_k (min %.6g)" % (label, Q.min())
    assert np.allclose(Q.sum(axis=1), F[:, -1], atol=1e-12), \
        "%s: q_k does not sum to F(7)" % label
    assert np.array_equal(H, (H > 0).astype(int)), "%s: h_k not binary" % label
    if "ons_obs" in df.columns:
        h_from_ons = np.column_stack(
            [(df.ons_obs.values == k - 1).astype(int) for k in range(1, KMAX + 1)])
        assert np.array_equal(H, h_from_ons), "%s: h_k disagrees with ons_obs" % label

    # --- reproducibility check against the archived lead-time metrics ---
    max_err = 0.0
    if ref is not None:
        for k in range(1, KMAX + 1):
            s = score(Y[:, k - 1], F[:, k - 1])
            r = ref[(ref.lead == k)].iloc[0] if "lead" in ref.columns else \
                ref[(ref.horizon == k) & (ref.track == "raw")].iloc[0]
            max_err = max(max_err, max(abs(s[c] - r[c]) for c in ("base", "brier", "bss", "auc")))
        assert max_err < TOL, "%s: cannot reproduce archived metrics (err %.3g)" % (label, max_err)

    sel = np.ones(n_all, bool) if holdout_mask is None else np.asarray(holdout_mask, bool)
    rows = []
    for k in range(1, KMAX + 1):
        hk, qk, fk, yk = H[sel, k - 1], Q[sel, k - 1], F[sel, k - 1], Y[sel, k - 1]
        m, c = score(hk, qk), score(yk, fk)
        rows.append(dict(track=label, day=k, n=int(sel.sum()),
                         n_pos_day=int(hk.sum()), base_day=m["base"],
                         q_mean=float(qk.mean()), brier_day=m["brier"],
                         bss_day=m["bss"], auc_day=m["auc"],
                         n_pos_cum=int(yk.sum()), base_cum=c["base"],
                         brier_cum=c["brier"], bss_cum=c["bss"], auc_cum=c["auc"]))
    return pd.DataFrame(rows), max_err


def main():
    # ---- GFS real-forecast track (raw; 455 issues, 2024-2026) ----
    g = pd.read_csv(os.path.join(RES, "gfs_hindcast_cases.csv"))
    gl = pd.read_csv(os.path.join(RES, "gfs_leadtime.csv"))
    g_tab, g_err = load_track(g, "GFS", "p_d", ref=gl)

    # ---- Perfect-prog H=7 clean track (raw; 437 held-out issues) ----
    h = pd.read_csv(os.path.join(HERE, "P1_A6_h7_hindcast_cases.csv"))
    hl = pd.read_csv(os.path.join(HERE, "P1_A6_h7_leadtime.csv"))
    h_tab, h_err = load_track(h, "PP_holdout", "p_d",
                              holdout_mask=(h.year >= 2024).values, ref=None)
    # self-check on the full pooled track against its own reference table
    _, h_err_pool = load_track(h, "PP_pooled", "p_d", ref=hl)

    out = pd.concat([g_tab, h_tab], ignore_index=True)
    out.to_csv(OUT_CSV, index=False, float_format="%.6f")
    print("wrote", OUT_CSV, "rows =", len(out))
    print("reproducibility max error: GFS %.2e | PP pooled %.2e" % (g_err, h_err_pool))
    print("\nGFS: cumulative BSS  %.3f -> %.3f (rises)" %
          (g_tab.bss_cum.iloc[0], g_tab.bss_cum.iloc[-1]))
    print("GFS: day-specific BSS %.3f -> %.3f | AUC %.3f -> %.3f (decays)" %
          (g_tab.bss_day.iloc[0], g_tab.bss_day.iloc[-1],
           g_tab.auc_day.iloc[0], g_tab.auc_day.iloc[-1]))
    print("PP holdout: day-specific BSS %.3f -> %.3f | AUC %.3f -> %.3f" %
          (h_tab.bss_day.iloc[0], h_tab.bss_day.iloc[-1],
           h_tab.auc_day.iloc[0], h_tab.auc_day.iloc[-1]))

    # ---- ready-to-paste markdown table (3 decimals, matching Tables 6/8a) ----
    def f3(x, sign=False):
        """Round half up at 3 decimals, matching Tables 6/8a (not bankers' rounding)."""
        # quantise the binary float back to the CSV's 6-decimal precision first,
        # otherwise -0.3415 arrives as -0.34149999... and rounds the wrong way.
        q = Decimal(str(round(float(x), 6))).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP)
        s = ("%+.3f" % q) if sign else ("%.3f" % q)
        # the manuscript uses a mathematical minus sign throughout (Tables 6/8a)
        return s.replace("-", "\u2212")

    lines = []
    for lbl, t in [("A", g_tab), ("B", h_tab)]:
        lines.append("Panel %s — %s" % (lbl,
                     "GFS real-forecast track, 455 issues (raw ensemble frequencies)"
                     if lbl == "A" else
                     "Perfect-prog H = 7 track, 437 held-out issues (raw ensemble frequencies)"))
        lines.append("")
        lines.append("| k | First crossings | Base rate | BSS, day k | AUC, day k |")
        lines.append("|---:|---:|---:|---:|---:|")
        for _, r in t.iterrows():
            lines.append("| %d | %d | %s | %s | %s |" %
                         (r.day, r.n_pos_day, f3(r.base_day),
                          f3(r.bss_day, True), f3(r.auc_day)))
        lines.append("")
    table_md = "\n".join(lines)

    md = io.open(OUT_MD, "w", encoding="utf-8", newline="\n")
    md.write("""# B1 — Day-specific (first-passage-day) verification

**Generated**: `work/analysis/p1_a_b1_day_verification.py` (no re-fitting, no new data).
**Numeric output**: `work/results/paperA_b1_day_verification.csv`.

## What is computed

The archived verification uses a cumulative-window predictand
$y_{dk}=1\\{C^*\\text{ crossed within }k\\text{ days after issue}\\}$ and the ensemble
frequency $p_{dk}=P(\\text{first crossing within }k\\text{ days})$. Both are monotone
in $k$, so the base rate grows with nominal lead (GFS 0.022 → 0.136) and the
cumulative BSS rises (0.176 → 0.437) even though the underlying forecast does
not improve. This is the effect queried in major comment 3.

The exact algebraic decomposition

$$q_k=p_{dk}-p_{d(k-1)},\\qquad h_k=y_{dk}-y_{d(k-1)},\\qquad p_{d0}=y_{d0}=0$$

gives the discrete-time **first-passage-day (hazard)** predictand: $h_k=1$ when
the first crossing falls on day $k$, and $q_k$ is the archived ensemble frequency
of exactly that event. For the perfect-prog H = 7 track, $h_k$ was checked
cell by cell against the archived `ons_obs` column (identical). Scoring $q_k$
against $h_k$ separately at each $k$ removes the accumulating base rate, so the
genuine lead dependence becomes visible. BSS and AUC follow the same definitions
as `gfs_leadtime.csv` / `P1_A6_h7_leadtime.csv`: BSS against the climatological
base rate of the scored sample, AUC by `roc_auc_score`.

## Why only raw frequencies

The calibrated columns `pc_dk` are produced by one isotonic map **per horizon**,
fitted independently; they are not monotone across $k$ (166 negative differences,
minimum −0.600 in the archived H = 7 cases), so differencing them would yield
invalid probabilities. The raw $p_{dk}$ are cumulative by construction and their
differences are non-negative (asserted: 0 negative cells). No isotonic map,
ensemble run or random draw is introduced here. AUC is nearly but not exactly
invariant to isotonic recalibration because the maps create ties: the raw
cumulative AUC at $k=3$ (0.999) differs slightly from the calibrated value
tabulated in Table 6 (0.996). Table 12 therefore reports **day-k BSS and AUC
only**; cumulative values are not re-tabulated. (Panel A's cumulative BSS would
duplicate Table 8a exactly, but Panel B's differs from Table 6 in the last digit
at $k=3$ and $k=5$ — 0.092 vs 0.093 and −0.341 vs −0.342 — because the archived
per-issue probabilities are stored at four decimals whereas Table 6 was
recomputed on a higher-precision path. The contrast is drawn in the text
against Tables 8a and 6 instead.)

## Reproducibility self-check (asserted in the script)

Recomputing the archived cumulative metrics from the per-issue files reproduces
`gfs_leadtime.csv` (7 leads) and `P1_A6_h7_leadtime.csv` (7 leads × raw track)
to a maximum absolute error of %.2e and %.2e respectively — pure 4-decimal
rounding. The script aborts if this check fails.

## Result

* **GFS**: cumulative-window BSS rises 0.176 → 0.437, but day-specific BSS falls
  0.176 → 0.052 and day-specific AUC falls 0.931 → 0.868. The apparent
  "longer lead is better" shape is a base-rate artefact of the window union.
* **Perfect-prog held-out**: cumulative raw BSS falls 0.431 → −0.745 (mainly
  accumulating overconfidence), while day-specific BSS falls 0.431 → −0.003 and
  day-specific AUC falls 1.000 → 0.925. Discrimination decays with lead, as
  expected; the raw cumulative collapse is a reliability effect, not a loss of
  ranking.

## Limits of this table

* Each day has only 7–10 positives (GFS 8–10, held-out perfect-prog 7–9); no
  bootstrap interval is attached and the day-specific BSS values are indicative,
  not precise skill estimates.
* The predictand is **first crossing on day k**, not "red state on day k".
  A day-specific red-state label paired with a day-specific red-state probability
  would require the per-day ensemble frequencies `p_day`, which were not archived;
  regenerating them needs the missing GFS generator (B6).
* Both tracks remain self-referential: labels come from the index being forecast.
  This table changes the *shape* of the verification, not its independence.

## Table 12 (ready to paste; 3 decimals, matching Tables 6 and 8a)

""" % (g_err, h_err_pool))
    md.write(table_md)
    md.write("\n\nSource rows: `work/results/paperA_b1_day_verification.csv`. "
             "`bss_day` and `auc_day` are new; `bss_cum` / `auc_cum` are kept in "
             "the CSV for the reproducibility self-check but are not printed in "
             "Table 12. Three-decimal values use round-half-up after quantising "
             "the binary float to the CSV's six-decimal precision.\n")
    md.close()
    print("wrote", OUT_MD)
    print("\n--- Table 12 preview ---\n" + table_md)


if __name__ == "__main__":
    main()
