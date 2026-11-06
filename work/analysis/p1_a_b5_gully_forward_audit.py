# -*- coding: utf-8 -*-
"""B5: 84-gully inventory as an external anchor — feasibility audit.

Reviewer Major 2 asks for a *forward* test: "use the historical event date of
each gully in the inventory to test whether the CTI is elevated on event days
(even if only the nine large/extreme events)".  The hand-over note of
2026-11-06 asserted that `work/clean/gullies_features.csv` "does contain an
event-date field (e.g. 2015-05-08, 2020-07-25)".  That assertion is wrong.

This script makes the contrary verdict reproducible instead of rhetorical,
and computes the one diagnostic that the data do support: the seasonality of
the remote-sensing acquisition dates against the CTI red-state season.  It is
run with the Anaconda interpreter (pandas).  No number is invented: every
row of both output CSVs is computed from the archived files listed below.

Inputs
------
  work/clean/gullies_features.csv             84 x 81 cleaned attribute table
  work/data/_（6）.../*.shp.xml               ArcGIS lineage / field metadata
  work/tmp/gully_cols.txt                     original DBF field-name list
  work/results/p2_cti_series.csv              1,944 daily CTI values

Outputs
-------
  work/results/paperA_b5_inventory_audit.csv
  work/results/paperA_b5_image_seasonality.csv
  work/analysis/B5_gully_forward_test.md

Verdict (see the assertions below): the requested forward test cannot be run
on these data.  Two independent reasons: (i) the two date-like columns are
remote-sensing interpretation / image-acquisition metadata, with 1 and 7
distinct values for 84 gullies, not 84 event dates; (ii) all eight distinct
dates lie in 2010-2020, entirely before the CTI series starts on 2021-05-01.
"""
import glob
import io
import os
import re
import struct

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLEAN = os.path.join(ROOT, "work", "clean")
RES = os.path.join(ROOT, "work", "results")
DATA = os.path.join(ROOT, "work", "data")

GULLY = os.path.join(CLEAN, "gullies_features.csv")
CTI = os.path.join(RES, "p2_cti_series.csv")
CSTAR = 2.5
EXT_EVENTS = ["2026-08-26", "2025-07-08"]   # the only dated external events in the paper


def f(x, nd=3):
    return ("%." + str(nd) + "f") % x


# ---------------------------------------------------------------- load inputs
g = pd.read_csv(GULLY, encoding="utf-8-sig")
s = pd.read_csv(CTI, parse_dates=["date"])
assert "CTI" in s.columns and "date" in s.columns

xml_files = glob.glob(os.path.join(DATA, "_（6）*", "*", "*.shp.xml"))
assert len(xml_files) == 1, "expected exactly one inventory .shp.xml"
xml = io.open(xml_files[0], encoding="utf-8", errors="replace").read()
dbf_files = glob.glob(os.path.join(DATA, "_（6）*", "*", "*.dbf"))
assert len(dbf_files) == 1, "expected exactly one inventory .dbf"
dbf = open(dbf_files[0], "rb").read()
cols_txt = io.open(os.path.join(ROOT, "work", "tmp", "gully_cols.txt"),
                   encoding="utf-8", errors="replace").read()

# ------------------------------------------------- A1: inventory shape & dates
assert g.shape == (84, 81), "A1: inventory shape changed: %s" % (g.shape,)
date_cols = []
for c in g.columns:
    v = g[c].dropna().astype(str)
    if len(v) and (v.str.match(r"^\d{4}-\d{2}-\d{2}$").mean() > 0.9):
        date_cols.append(c)
assert set(date_cols) == {"解译Time", "影像Time"}, (
    "A1: unexpected date columns %s" % date_cols)
print("A1 OK  inventory 84 x 81; date-like columns =", date_cols)

# -------------------------------------- A2/A3: the dates are imaging metadata
interp = g["解译Time"].astype(str)
image = g["影像Time"].astype(str)
n_interp = interp.nunique()
n_image = image.nunique()
assert n_interp == 1 and interp.iloc[0] == "2020-07-25", (
    "A3: 解译Time is not the single batch interpretation date")
assert n_image == 7, "A3: expected 7 image-acquisition batches, got %d" % n_image
assert g["影像类"].nunique() == 1 and g["影像type"].nunique() == 1, (
    "A3: imaging platform/type should be constant")
print("A3 OK  解译Time: 1 value (84/84); 影像Time: 7 batches; 影像类/影像type constant")

# ------------------------------------------ A4: ArcGIS lineage proves intent
calc_time = re.findall(r'CalculateField[^\"]*?影像Time\s*\"([^\"]+)\"', xml)
calc_interp = re.findall(r'CalculateField[^\"]*?解译Time\s*\"([^\"]+)\"', xml)
assert len(calc_time) >= 4 and len(calc_interp) >= 1, (
    "A4: expected batch CalculateField assignments for the date fields")
# the attribute table carries no event-date field at all
n_rec = struct.unpack("<I", dbf[4:8])[0]
h_len = struct.unpack("<H", dbf[8:10])[0]
r_len = struct.unpack("<H", dbf[10:12])[0]
n_fld = (h_len - 33) // 32
dbf_names = []
for i in range(n_fld):
    o = 33 + 32 * i + 1          # +1: this DBF prefixes one pad byte
    dbf_names.append(dbf[o:o + 11].split(b"\x00")[0].decode("gbk", "replace").strip())
joined = " ".join(dbf_names) + " " + cols_txt + " " + " ".join(g.columns)
for kw in ["发生时间", "事件日期", "发灾", "event_date", "occurrence"]:
    assert kw not in joined, "A4: an event-date field exists after all: %s" % kw
assert n_rec == 84, "A4: DBF record count %d" % n_rec
print("A4 OK  DBF %d records / %d fields, no event-date field; "
      "%d 影像Time and %d 解译Time CalculateField batch assignments in .shp.xml"
      % (n_rec, n_fld, len(calc_time), len(calc_interp)))

# --------------------------------------------- A5: no temporal overlap at all
all_dates = sorted(set(interp) | set(image))
cti_start, cti_end = s.date.min(), s.date.max()
n_overlap = sum(cti_start <= pd.Timestamp(d) <= cti_end for d in all_dates)
assert n_overlap == 0, "A5: some inventory date falls inside the CTI series"
assert len(all_dates) == 8, "A5: expected 8 distinct dates, got %d" % len(all_dates)
print("A5 OK  8 distinct dates all in 2010-2020; intersection with CTI "
      "(%s .. %s) is empty" % (cti_start.date(), cti_end.date()))

# ---------------------------------------- A6: reproduce the red-day catalogue
red = s[s.CTI >= CSTAR].copy()
assert len(red) == 38, "A6: expected 38 red days, got %d" % len(red)
red_months = red.date.dt.month.value_counts().sort_index().to_dict()
assert set(red_months) <= {6, 7, 8, 9}, (
    "A6: red days outside June-September: %s" % red_months)
assert len(s) == 1944, "A6: CTI series length changed"
print("A6 OK  %d red days of %d CTI days; month counts %s"
      % (len(red), len(s), red_months))

# the two external events carry the CTI values quoted in the manuscript
ev_cti = {}
for d in EXT_EVENTS:
    row = s[s.date == pd.Timestamp(d)]
    assert len(row) == 1, "A6: event %s absent from CTI series" % d
    ev_cti[d] = float(row.CTI.iloc[0])
assert abs(ev_cti["2026-08-26"] - 2.73) < 0.01, ev_cti
assert abs(ev_cti["2025-07-08"] - 2.69) < 0.01, ev_cti
print("A6 OK  external event CTI reproduced:",
      {k: round(v, 3) for k, v in ev_cti.items()})

# ------------------------------------- A7: image batches miss the red season
def season_stats(dates, label):
    rows = []
    for d in dates:
        dt = pd.Timestamp(d)
        doy = int(dt.dayofyear)
        # day-of-year window +-7 d across all years of the CTI series
        dd = s.date.dt.dayofyear
        win = s[(dd >= doy - 7) & (dd <= doy + 7)]
        n_red = int((win.CTI >= CSTAR).sum())
        rows.append(dict(kind=label, date=d, year=int(dt.year), doy=doy,
                         month=int(dt.month), n_cti_days_in_window=len(win),
                         n_red_in_window=n_red,
                         in_red_season=bool(6 <= dt.month <= 9)))
    return rows


season = season_stats(sorted(image.unique()), "image batch")
season += season_stats(sorted(interp.unique()), "interpretation batch")
sea = pd.DataFrame(season).sort_values(["kind", "date"]).reset_index(drop=True)
img = sea[sea.kind == "image batch"]
assert len(img) == 7 and int(img.in_red_season.sum()) == 0, (
    "A7: an image batch falls inside the CTI red season")
assert int(img.n_red_in_window.sum()) == 0, (
    "A7: a +-7 d window around an image batch contains a red day")
print("A7 OK  7/7 image batches outside June-September; "
      "0 red days in any +-7 d day-of-year window")

# the only date inside the red season is the interpretation batch, which is a
# mapping date, not an event date
assert int(sea[sea.kind == "interpretation batch"].in_red_season.iloc[0]) == 1
print("    (the single in-season date is 2020-07-25, the mapping batch)")

# ------------------------------------------------------------ write CSV: audit
audit = [
    dict(check="Inventory records", observed="84 gullies, 81 columns",
         required_for_forward_test="one dated event per gully",
         verdict="meta", evidence="gullies_features.csv"),
    dict(check="Date-like columns", observed="解译Time, 影像Time (2 of 81)",
         required_for_forward_test="an event-occurrence date column",
         verdict="FAIL", evidence="column scan, ISO-date regex >0.9"),
    dict(check="解译Time distinct values", observed="1 (2020-07-25, 84/84 rows)",
         required_for_forward_test="up to 84 distinct event dates",
         verdict="FAIL", evidence="value_counts"),
    dict(check="影像Time distinct values", observed="7 (2010-05-08 .. 2017-11-11)",
         required_for_forward_test="up to 84 distinct event dates",
         verdict="FAIL", evidence="value_counts"),
    dict(check="Imaging platform / type", observed="Google earth / 1 m, constant",
         required_for_forward_test="not applicable to event dating",
         verdict="meta", evidence="影像类, 影像type"),
    dict(check="ArcGIS lineage", observed="%d + %d CalculateField batch assignments"
         % (len(calc_time), len(calc_interp)),
         required_for_forward_test="a recorded event chronology",
         verdict="FAIL", evidence=".shp.xml Process Step records"),
    dict(check="Original DBF fields", observed="%d fields, no event-date field" % n_fld,
         required_for_forward_test="an event-date field",
         verdict="FAIL", evidence="DBF header, gully_cols.txt"),
    dict(check="Dates inside CTI series", observed="0 of 8 (CTI starts %s)"
         % cti_start.date(),
         required_for_forward_test=">=1", verdict="FAIL",
         evidence="p2_cti_series.csv date range"),
    dict(check="Dated external events", observed="2 (2026-08-26, 2025-07-08), "
         "both post-selected", required_for_forward_test="an event sample",
         verdict="PARTIAL", evidence="Sections 4.1, 4.7"),
    dict(check="Red days / CTI days", observed="38 / 1944 (June-September only)",
         required_for_forward_test="event dates inside the series",
         verdict="meta", evidence="p2_cti_series.csv, CTI >= 2.5"),
    dict(check="Image batches in red season", observed="0 of 7",
         required_for_forward_test="n/a (detection-bias diagnostic)",
         verdict="meta", evidence="paperA_b5_image_seasonality.csv"),
]
aud = pd.DataFrame(audit)
aud.to_csv(os.path.join(RES, "paperA_b5_inventory_audit.csv"),
           index=False, encoding="utf-8-sig")
sea.to_csv(os.path.join(RES, "paperA_b5_image_seasonality.csv"),
           index=False, encoding="utf-8-sig")

# ---------------------------------------------------------- write methods note
md = []
md.append("# B5 — 84-gully inventory as an external anchor: feasibility audit\n")
md.append("Run: `C:/anaconda3/python.exe work/analysis/p1_a_b5_gully_forward_audit.py`.\n"
          "Inputs: `work/clean/gullies_features.csv`, the original `（6）历史灾点数据` "
          "`.dbf`/`.shp.xml`, `work/tmp/gully_cols.txt`, "
          "`work/results/p2_cti_series.csv`. Outputs: "
          "`work/results/paperA_b5_inventory_audit.csv` (%d rows) and "
          "`work/results/paperA_b5_image_seasonality.csv` (%d rows).\n"
          % (len(aud), len(sea)))
md.append("## Verdict\n")
md.append("The forward test requested in reviewer Major 2 — using each gully's "
          "historical **event date** to test whether the CTI is elevated on "
          "those days — **cannot be computed from the archived data**. This is "
          "a data-availability verdict, not a negative scientific result, and "
          "it must not be reported as a test that was run.\n")
md.append("Two independent blockers:\n")
md.append("1. **No event-date field exists.** Of the 81 columns, exactly two "
          "are ISO dates: `解译Time` (interpretation date) and `影像Time` "
          "(image-acquisition date). `解译Time` takes **one** value for all 84 "
          "gullies (2020-07-25); `影像Time` takes **seven** values "
          "(2010-05-08 to 2017-11-11). The original DBF has %d fields and "
          "contains no occurrence/date field. The `.shp.xml` ArcGIS lineage "
          "shows these columns were populated by `CalculateField` batch "
          "assignments (e.g. `影像Time \"2015/5/08\"`), i.e. they are "
          "mapping metadata for a remote-sensing interpretation, not a "
          "chronology of debris-flow events.\n" % n_fld)
md.append("2. **No temporal overlap.** All eight distinct dates fall in "
          "2010-2020, while the CTI series begins %s. Even if the imaging "
          "dates were misread as event dates, there is no CTI value to look up "
          "for any of them. The station record used to build the marginals "
          "starts 2021-01-01; the paper's only dated external events are "
          "%s (CTI %s) and %s (CTI %s), both selected after they occurred.\n"
          % (cti_start.date(), EXT_EVENTS[0], f(ev_cti[EXT_EVENTS[0]], 2),
             EXT_EVENTS[1], f(ev_cti[EXT_EVENTS[1]], 2)))
md.append("The manuscript already states that the catchment has *no dated "
          "landslide inventory* (Section 5.7, comparator limitation); this "
          "audit makes that statement verifiable and corrects the contrary "
          "note in `revision_unfinished_2026-11-06.md` (B5).\n")
md.append("## What can be computed: image-acquisition seasonality\n")
md.append("The %d CTI days contain %d red days (CTI >= %s), all in "
          "June-September (month counts %s). All seven image-acquisition "
          "batches lie outside that season: %s. A day-of-year window of +-7 d "
          "around each batch contains **0** red days in every case. The single "
          "date inside the red season is 2020-07-25, which is the "
          "*interpretation* batch, not an image or event date.\n"
          % (len(s), len(red), CSTAR, red_months,
             ", ".join(sorted(image.unique()))))
md.append("This is a **descriptive** diagnostic with seven batches; it is not "
          "a significance test and cannot be one. Remote-sensing acquisition "
          "is itself seasonally chosen (clear-sky windows), so a null model of "
          "uniform dates is not defensible. What it does establish is that the "
          "imagery underlying the inventory was not acquired during the "
          "season in which the CTI is ever red, which is a concrete, "
          "quantified form of the detection bias already acknowledged in "
          "Section 4.1 (reviewer Minor 5).\n")
md.append("## Table 14 (draft)\n")
md.append("**Panel A — audit of the inventory as a dated event sample**\n")
md.append("| Check | Observed | Verdict |")
md.append("|---|---|:--:|")
pick = ["Date-like columns", "解译Time distinct values", "影像Time distinct values",
        "ArcGIS lineage", "Original DBF fields", "Dates inside CTI series",
        "Dated external events"]
for _, r in aud[aud.check.isin(pick)].iterrows():
    md.append("| %s | %s | %s |" % (r["check"], r["observed"], r["verdict"]))
md.append("\n**Panel B — image batches against the CTI red season**\n")
md.append("| Image date | Month | DOY | CTI days in +-7 d DOY window | "
          "Red days in window | In red season (Jun-Sep) |")
md.append("|---|---:|---:|---:|---:|:--:|")
for _, r in img.iterrows():
    md.append("| %s | %d | %d | %d | %d | %s |"
              % (r.date, r.month, r.doy, r.n_cti_days_in_window,
                 r.n_red_in_window, "yes" if r.in_red_season else "no"))
md.append("\n## Limits and what would be needed\n")
md.append("- A genuine forward test requires an event catalogue with occurrence "
          "dates inside 2021-2026, or a station-independent reanalysis CTI "
          "extending back to 2010-2020 **plus** dated events. Neither exists "
          "in the archived inputs; the two are separate gaps.\n")
md.append("- `work/data/meteo/era5grid/` and `work/data/power/` do hold daily "
          "2005-2025 series, so a station-independent CTI proxy is in principle "
          "reconstructible (that is B4, still outstanding). It would not by "
          "itself supply event dates.\n")
md.append("- No number in this note is a skill score, p value or event-level "
          "validation. The external evidence in the paper remains two "
          "post-selected events plus an undated structural stratification.\n")

io.open(os.path.join(ROOT, "work", "analysis", "B5_gully_forward_test.md"),
        "w", encoding="utf-8").write("\n".join(md) + "\n")

print("\n=== Table 14 (draft) ===")
_i = next(i for i, x in enumerate(md) if x.startswith("## Table 14"))
_j = next(i for i, x in enumerate(md) if x.lstrip().startswith("## Limits"))
print("\n".join(md[_i:_j]))
print("\nWROTE paperA_b5_inventory_audit.csv (%d rows), "
      "paperA_b5_image_seasonality.csv (%d rows), B5_gully_forward_test.md"
      % (len(aud), len(sea)))
