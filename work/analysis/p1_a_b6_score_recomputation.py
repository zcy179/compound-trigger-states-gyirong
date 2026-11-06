# -*- coding: utf-8 -*-
"""B6 (review major point 5): data-level reproducibility of the GFS track.

READ-ONLY. This script does NOT generate, refit, resample or otherwise create
any forecast probability. It reads the archived per-issue probabilities and
labels and recomputes the *scores* that were reported, then asserts that they
match the archived score tables. It writes a single audit CSV.

Why this exists
---------------
The script that turned raw GFS fields into the 455-issue 1,500-member ensemble
probabilities was not preserved, and the stochastic runs recorded no random
seed. Re-running that generator would not reproduce the reported numbers bit
for bit. NHESS's data policy instead asks that "the data needed to replicate
figures in a paper" be publicly available, in a database or in a supplement.
Those data are the 455 per-issue probabilities and labels, which are archived.

What IS recomputable (asserted below)
------------------------------------
  A. Table 8a / gfs_leadtime.csv : BSS, AUC, base rate, positives, Brier,
     for D = 1..7 from p_dk and y_dk of gfs_hindcast_cases.csv.
  B. Supplementary Table S11 Panel A : day-k (first-passage) BSS and AUC from
     the exact algebraic decomposition q_k = F_k - F_{k-1}, h_k = z_k - z_{k-1}.
  C. The GFS curve of Fig. 5a/5b is the same seven BSS/AUC values as A; it is
     listed as identical-by-construction, not counted twice.

What is NOT recomputable (declared, never asserted)
--------------------------------------------------
  * the 455 per-issue probabilities p_dk themselves (no generator, no seed);
  * the 10-day block-bootstrap 95% intervals of Table 8a (no seed);
  * the GFS side of the 437-issue paired comparison (bss_gfs 0.3327,
    auc_gfs 0.9502, and hence the differences 0.4025 / 0.0481). No archived
    column combination of gfs_hindcast_cases.csv and p3_hindcast_cases.csv
    reproduces these values; their generating chain is not located. The
    perfect-prog side (bss_pp 0.7366) IS reproducible from an isotonic map on
    2021-2023, but that is outside the B6 GFS scope and is not claimed here.

No number in the manuscript is changed by this script.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(ROOT, "work", "results")
OUT = os.path.join(RES, "paperA_b6_score_recomputation.csv")

TOL_LEAD = 5e-5   # gfs_leadtime.csv stores 4 decimals
TOL_DAY = 1e-5    # paperA_b1_day_verification.csv stores 6 decimals

sys.stdout.reconfigure(encoding="utf-8")


def bss_of(p, y):
    """Brier score, climatological Brier, Brier skill score."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    bs = float(np.mean((p - y) ** 2))
    clim = float(np.mean((y.mean() - y) ** 2))
    return bs, clim, 1.0 - bs / clim


def main():
    cases = pd.read_csv(os.path.join(RES, "gfs_hindcast_cases.csv"))
    lead = pd.read_csv(os.path.join(RES, "gfs_leadtime.csv"))
    s11 = pd.read_csv(os.path.join(RES, "paperA_b1_day_verification.csv"))
    pair = pd.read_csv(os.path.join(RES, "p3_paired_pp_gfs.csv")).iloc[0]

    assert len(cases) == 455, "expected 455 archived GFS issues"
    assert len(lead) == 7, "expected 7 lead rows"
    assert all(f"p_d{k}" in cases and f"y_d{k}" in cases for k in range(1, 8))

    rows = []

    def add(scope, day, quantity, recomputed, archived, tol, status, reason=""):
        diff = abs(float(recomputed) - float(archived))
        if status == "RECOMPUTED_MATCH":
            assert diff <= tol, (
                f"{scope} D={day} {quantity}: recomputed {recomputed:.8f} "
                f"vs archived {archived:.8f} (diff {diff:.2e} > tol {tol:.1e})"
            )
        rows.append(dict(scope=scope, day=day, quantity=quantity,
                         recomputed=round(float(recomputed), 8),
                         archived=round(float(archived), 8),
                         abs_diff=round(diff, 10), status=status, reason=reason))

    # ---- A. Table 8a / gfs_leadtime.csv -----------------------------------
    for k in range(1, 8):
        p = cases[f"p_d{k}"].values
        y = cases[f"y_d{k}"].values
        bs, clim, bss = bss_of(p, y)
        auc = roc_auc_score(y, p)
        r = lead.loc[lead.lead == k].iloc[0]
        add("Table 8a / Fig. 5", k, "BSS", bss, r.bss, TOL_LEAD, "RECOMPUTED_MATCH")
        add("Table 8a / Fig. 5", k, "AUC", auc, r.auc, TOL_LEAD, "RECOMPUTED_MATCH")
        add("Table 8a / Fig. 5", k, "base_rate", y.mean(), r.base, TOL_LEAD, "RECOMPUTED_MATCH")
        add("Table 8a / Fig. 5", k, "positives", y.sum(), r.pos, 0, "RECOMPUTED_MATCH")
        add("Table 8a / Fig. 5", k, "brier", bs, r.brier, TOL_LEAD, "RECOMPUTED_MATCH")
        add("Table 8a / Fig. 5", k, "n_issues", len(y), r.n, 0, "RECOMPUTED_MATCH")

    # ---- B. Supplementary Table S11 Panel A (GFS day-k decomposition) -----
    g11 = s11[s11.track == "GFS"].sort_values("day").reset_index(drop=True)
    assert len(g11) == 7, "expected 7 GFS day rows in the S11 archive"
    prev_p = np.zeros(len(cases))
    prev_y = np.zeros(len(cases))
    for k in range(1, 8):
        Fk = cases[f"p_d{k}"].values
        zk = cases[f"y_d{k}"].values
        q = Fk - prev_p          # first-crossing forecast for day k
        h = zk - prev_y          # first-crossing event on day k
        assert np.all(h >= 0) and set(np.unique(h)) <= {0.0, 1.0}, \
            f"day {k}: h_k is not a valid 0/1 first-crossing label"
        _, _, bss_d = bss_of(q, h)
        auc_d = roc_auc_score(h, q)
        r = g11.loc[g11.day == k].iloc[0]
        add("Suppl. Table S11 Panel A", k, "BSS_day", bss_d, r.bss_day, TOL_DAY, "RECOMPUTED_MATCH")
        add("Suppl. Table S11 Panel A", k, "AUC_day", auc_d, r.auc_day, TOL_DAY, "RECOMPUTED_MATCH")
        add("Suppl. Table S11 Panel A", k, "first_crossings", h.sum(), r.n_pos_day, 0, "RECOMPUTED_MATCH")
        add("Suppl. Table S11 Panel A", k, "base_rate_day", h.mean(), r.base_day, TOL_DAY, "RECOMPUTED_MATCH")
        prev_p, prev_y = Fk, zk

    # ---- C. Declared non-recomputable quantities --------------------------
    not_recomputable = [
        ("Table 8a", "all", "block-bootstrap 95% CI (14 bounds)",
         "The stochastic bootstrap recorded no random seed; the archived "
         "intervals are reported as published values, not recomputed."),
        ("GFS hindcast", "all", "per-issue probabilities p_d1..p_d7",
         "Upstream 1,500-member ensemble generator not preserved and no seed "
         "recorded; the archived probabilities are the input, not an output, "
         "of this script."),
        ("Sec. 4.5 paired", 437, "GFS BSS 0.3327",
         "No archived column combination of gfs_hindcast_cases.csv and "
         "p3_hindcast_cases.csv reproduces this value; generating chain not located."),
        ("Sec. 4.5 paired", 437, "GFS AUC 0.9502",
         "Same as above; the paired GFS side cannot be recomputed from the archive."),
        ("Sec. 4.5 paired", 437, "BSS difference 0.4025 / AUC difference 0.0481",
         "Inherited from the two non-recomputable paired values above."),
    ]
    for scope, day, quantity, reason in not_recomputable:
        rows.append(dict(scope=scope, day=day, quantity=quantity,
                         recomputed=np.nan, archived=np.nan, abs_diff=np.nan,
                         status="NOT_RECOMPUTABLE", reason=reason))

    # cross-check: the paired sample size and its base rate ARE in the archive.
    # NOTE: the column named "clim" in p3_paired_pp_gfs.csv holds the base rate
    # (24/437 = 0.05492), NOT the climatological Brier (which would be
    # 0.05492 * 0.94508 = 0.05190). Verified against p3_hindcast_cases.any_obs.
    pp = pd.read_csv(os.path.join(RES, "p3_hindcast_cases.csv"))
    shared = pp[["issue", "any_obs"]].merge(cases[["issue"]], on="issue", how="inner")
    base_shared = float(shared.any_obs.mean())
    add("Sec. 4.5 paired", 437, "n_paired", len(shared), pair.n_paired, 0,
        "RECOMPUTED_MATCH")
    add("Sec. 4.5 paired", 437, 'base_rate (archived column "clim")',
        base_shared, pair.clim, TOL_LEAD, "RECOMPUTED_MATCH")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, float_format="%.8f")

    n_ok = int((out.status == "RECOMPUTED_MATCH").sum())
    n_no = int((out.status == "NOT_RECOMPUTABLE").sum())
    print(f"wrote {OUT}")
    print(f"asserted recomputations : {n_ok}")
    print(f"declared non-recomputable: {n_no}")
    print(f"max |recomputed - archived| among asserted rows: "
          f"{out.loc[out.status == 'RECOMPUTED_MATCH', 'abs_diff'].max():.2e}")
    print("No forecast probability was generated or modified.")


if __name__ == "__main__":
    main()
