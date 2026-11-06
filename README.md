# Reproducibility archive

Companion data and code for the manuscript *"Forecasting compound trigger states for a
high-mountain ice–rock avalanche–debris-flow chain: a copula–first-passage framework with
calibrated, episode-aware verification"*. Archive assembled 6 November 2026.

The archive contains 99 files (12 Python scripts/modules, 82 data files, this README, a
checksum manifest, a licence file, a `.gitattributes` and a `.gitignore`), about 3.3 MB. All paths are relative to the archive root; no installation
or configuration is needed.

## Layout

```
.
├── README.md
├── MANIFEST.csv              byte size and SHA-256 for every file
└── work/
    ├── code/                 shared CTI library and analysis scripts
    ├── analysis/             verification scripts and their per-case tables
    ├── results/              derived tables behind the manuscript's figures and tables
    ├── clean/                cleaned station and gully series
    └── tmp/                  DBF field-name list (read by the gully inventory audit)
```

The original `work/...` directory layout is preserved because the scripts derive the project
root from their own location (`dirname(__file__)/../..`). Unzip anywhere and run them from the
archive root; no paths need editing.

## Requirements

Python 3 with `numpy`, `pandas`, `scipy` and `scikit-learn`. `matplotlib` is needed only by
`work/code/23_gully_trigger_type.py`.

## Reproducing the reported scores

From the archive root:

```
python work/analysis/p1_a_b6_score_recomputation.py
python work/analysis/p1_a_paired_main_track.py
python work/analysis/p1_a_b1_day_verification.py
python work/analysis/p1_a_b2_information_matched.py
cd work/analysis && python P2_holdout_reanalysis.py
```

`P2_holdout_reanalysis.py` resolves its per-case table relative to the working directory, so it
must be launched from `work/analysis/`. The other scripts resolve paths from `__file__`.

`p1_a_b6_score_recomputation.py` reads the archived per-issue probabilities and labels and
asserts 72 published point scores of manuscript Table 8a, the GFS curves of Fig. 5 and
Supplementary Table S11. The largest deviation is 4.86 × 10⁻⁵, attributable to four-decimal
rounding. This is score-level, not probability-level, reproducibility: no forecast probability
is regenerated, refitted or resampled.

`p1_a_paired_main_track.py` rebuilds the paired comparison of Section 4.5 and the cost–loss
curves of Section 4.6 from a single archive chain — the main H = 7 track with the 19 August 2026
cut-off (`work/analysis/P1_A6_h7_hindcast_cases.csv`). It asserts the published point scores on
both sides (perfect-prog BSS 0.7667 and AUC 0.996; GFS 0.3327 and 0.9502; GEFS 0.8744 and
0.1350) and reproduces the archived GFS cost–loss table to within 4.9 × 10⁻⁵. Point estimates
and intervals are written to `work/results/p1_a_paired_main_track.csv`, the cost–loss rows to
`work/analysis/P1_A8_costloss_main_track.csv`. The intervals use a 10-day block bootstrap with
1,000 resamples and the seed fixed at 20261030, so they do reproduce.

## Scripts that require external source data

| Script | Additional input required |
|---|---|
| `work/analysis/p1_a_horizon7.py` | `work/data/meteo/fcst_port.csv` (Open-Meteo point forecast) |
| `work/analysis/p1_a_b5_gully_forward_audit.py` | the competition-supplied gully inventory `.dbf` and `.shp.xml` |
| `work/analysis/gefs_event_check.py` | the 31-member GEFS fields in `work/data/meteo/gefs/`; it loads `work/code/10_gefs_leadtime.py`, which is included |

These inputs are third-party source data and are not redistributed here. They are cited by
DOI/URL in the manuscript's Data availability section and Supplementary Table S1.

## Data notes

- The five files listed in Supplementary Table S13 are byte-identical to the correspondingly
  named files in `work/results/`; a single copy is provided here.
- The gully inventory contains no event-occurrence dates. The resulting data-availability
  assessment is recorded in `work/results/paperA_b5_inventory_audit.csv`.
- In `p3_paired_pp_gfs.csv`, the column named `clim` holds the base rate (24/437 = 0.05492),
  not the climatological Brier score.

## Reproducibility limits

The script that converted raw GFS fields into the 455-issue, 1,500-member ensemble
probabilities was not preserved, and the archived stochastic runs recorded no random seed.
Consequently the per-issue probabilities themselves, and the 10-day block-bootstrap 95%
intervals of Tables 8a and 8b, are reported as published values and cannot be regenerated bit
for bit. Everything downstream of the archived probabilities does reproduce: all point scores
of Table 8a, the GFS curves of Fig. 5, Supplementary Table S11, the paired comparison of
Section 4.5 including its GFS and GEFS sides, and the cost–loss curves of Section 4.6.

## Source data

Station and background datasets are available from the National Tibetan Plateau / Third Pole
Environment Data Center (TPDC) under their own terms and companion-citation requirements;
ERA5-Land from the Copernicus Climate Data Store under the Copernicus licence; GFS/GEFS from
the NOAA open-data archives. See the manuscript's Data availability section and Supplementary
Table S1 for DOIs, URLs and access dates. Competition-provided attachments are not included.

## File integrity

`MANIFEST.csv` lists the byte size and SHA-256 digest of every file in the archive (the
manifest itself is not listed, since it cannot contain its own digest).

## Licence

CC BY 4.0 (full text in `LICENSE`). Please cite this archive by its repository URL and release
tag, together with the associated manuscript.
