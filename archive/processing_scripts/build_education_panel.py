"""
Dataset 5 — Education: interpolate 3 census snapshots to annual panel 2012–2021.
Source: sources/education_snapshots_insee.csv (288 rows, years 2011/2016/2022).
Output: sources/education_panel_insee.csv (960 rows, years 2012–2021).
"""

import sys
import numpy as np
import pandas as pd

SRC  = "sources/education_snapshots_insee.csv"
OUT  = "sources/education_panel_insee.csv"
DATA_SOURCES = "DATA_SOURCES.md"

PANEL_YEARS = list(range(2012, 2022))   # 2012–2021 inclusive
SNAP_YEARS  = [2011, 2016, 2022]
SPOT_DEPS   = ["75", "15", "31"]        # Paris, Cantal, Haute-Garonne

# ── TASK 1 — load snapshots ───────────────────────────────────────────────────

snaps = pd.read_csv(SRC, sep=";", dtype={"dep_code": str})
print(f"Snapshots loaded: {snaps.shape}  years={sorted(snaps.year.unique())}")

# pivot to wide: index=dep_code, columns=year
wide = snaps.pivot(index="dep_code", columns="year", values="edu_share_sup")
assert wide.shape == (96, 3), f"Expected (96,3), got {wide.shape}"
assert list(wide.columns) == SNAP_YEARS, f"Expected years {SNAP_YEARS}"

# ── TASK 1 — linear interpolation ────────────────────────────────────────────

records = []
for dep, row in wide.iterrows():
    v2011, v2016, v2022 = row[2011], row[2016], row[2022]
    xp = np.array(SNAP_YEARS, dtype=float)
    fp = np.array([v2011, v2016, v2022], dtype=float)

    for yr in PANEL_YEARS:
        val = float(np.interp(yr, xp, fp))
        val = round(val, 2)
        is_interp = yr != 2016
        records.append({
            "dep_code": dep,
            "year": yr,
            "edu_share_sup": val,
            "edu_is_interpolated": is_interp,
        })

panel = pd.DataFrame(records)
panel["dep_code"] = panel["dep_code"].astype(str)
panel["year"]     = panel["year"].astype(int)
panel = panel.sort_values(["dep_code", "year"]).reset_index(drop=True)

# ── TASK 2 — checks ───────────────────────────────────────────────────────────

print(f"\n── Shape checks ──")
print(f"  Rows        : {len(panel)} (expected 960)")
print(f"  Nulls       : {panel.isnull().sum().to_dict()}")
dup = panel.duplicated(subset=["dep_code", "year"]).sum()
print(f"  Dup keys    : {dup} (expected 0)")
per_dep = panel.groupby("dep_code").size()
print(f"  Rows/dep    : min={per_dep.min()} max={per_dep.max()} (expected 10 each)")
per_yr  = panel.groupby("year").size()
print(f"  Rows/year   : {per_yr.to_dict()} (each expected 96)")

# edu_is_interpolated flag check
n_obs  = (panel["edu_is_interpolated"] == False).sum()
n_intp = (panel["edu_is_interpolated"] == True).sum()
print(f"\n  edu_is_interpolated=False: {n_obs}  (expected 96 — year 2016 only)")
print(f"  edu_is_interpolated=True : {n_intp}  (expected 864)")

# Bounds [10,70]%
out_of_bounds = panel[(panel["edu_share_sup"] < 10) | (panel["edu_share_sup"] > 70)]
if len(out_of_bounds):
    print(f"\n  OUT-OF-BOUNDS rows:\n{out_of_bounds.to_string()}")
else:
    print(f"  All shares in [10, 70]% — OK")

# Bracket monotone check
print(f"\n── Bracket (monotone-safe) check ──")
outside = []
for dep, grp in panel.groupby("dep_code"):
    v2011 = wide.loc[dep, 2011]
    v2016 = wide.loc[dep, 2016]
    v2022 = wide.loc[dep, 2022]

    lo1, hi1 = min(v2011, v2016), max(v2011, v2016)
    lo2, hi2 = min(v2016, v2022), max(v2016, v2022)

    for _, r in grp.iterrows():
        yr  = r["year"]
        val = r["edu_share_sup"]
        if 2012 <= yr <= 2015:
            if not (lo1 - 0.005 <= val <= hi1 + 0.005):
                outside.append((dep, yr, val, lo1, hi1))
        elif 2017 <= yr <= 2021:
            if not (lo2 - 0.005 <= val <= hi2 + 0.005):
                outside.append((dep, yr, val, lo2, hi2))

if outside:
    print(f"  OUTSIDE BRACKET (should be none):")
    for dep, yr, val, lo, hi in outside:
        print(f"    dep={dep} yr={yr} val={val}  bracket=[{lo},{hi}]")
else:
    print(f"  All interpolated values within brackets — OK")

# ── Spot: sanity print for 3 departments ─────────────────────────────────────

print(f"\n── 10-year series for spot departments ──")
for dep in SPOT_DEPS:
    sub = panel[panel["dep_code"] == dep][["year", "edu_share_sup", "edu_is_interpolated"]]
    snap_row = wide.loc[dep]
    print(f"\n  dep={dep}  anchors: 2011={snap_row[2011]}%  2016={snap_row[2016]}%  2022={snap_row[2022]}%")
    print(sub.to_string(index=False))

# ── TASK 2 — write output ─────────────────────────────────────────────────────

panel.to_csv(
    OUT,
    sep=";",
    index=False,
    quoting=__import__("csv").QUOTE_NONNUMERIC,
)
print(f"\n── Output ──")
print(f"  Saved: {OUT}  ({len(panel)} rows × {len(panel.columns)} cols)")
check = pd.read_csv(OUT, sep=";", dtype={"dep_code": str})
print(f"  Re-read shape: {check.shape}")
print(f"  Cols: {check.columns.tolist()}")
print(f"  Head:\n{check.head(6).to_string(index=False)}")

# ── TASK 3 — append DATA_SOURCES.md ──────────────────────────────────────────

entry = """
---

## 7. Education by Department — Annual Panel 2012–2021 (Interpolated)

- **Source**: Derived from `sources/education_snapshots_insee.csv` (section 6 above)
- **Method**: Linear interpolation across three census anchor years (2011, 2016, 2022) using `numpy.interp`
  - 2012–2015: straight-line between 2011 and 2016 observed values
  - 2016: observed value carried through unchanged
  - 2017–2021: straight-line between 2016 and 2022 observed values
- **Coverage**: 96 metropolitan departments × 10 years = 960 rows
- **Variable**: `edu_share_sup` — higher-ed share (Bac+2 and above, %), rounded to 2 dp; definition consistent across all three anchor vintages
- **Flag**: `edu_is_interpolated` — `False` only for year 2016 (observed within the panel window); `True` for all other years (2012–2015, 2017–2021)
- **Output file**: `sources/education_panel_insee.csv` (960 rows × 4 cols: dep_code, year, edu_share_sup, edu_is_interpolated; `;`-delimited)
- **Note**: Only 2016 is a real within-panel observation; 2012–2015 and 2017–2021 are interpolated estimates. Do not over-interpret year-to-year variation.
"""

with open(DATA_SOURCES, "a", encoding="utf-8") as fh:
    fh.write(entry)
print(f"\n  Appended entry to {DATA_SOURCES}")
print("\nDone. STOP — merge is the next step.")
