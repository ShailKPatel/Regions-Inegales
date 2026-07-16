#!/usr/bin/env python3
"""
Verification script for firms_panel.csv.
Tasks: spot-check vs raw, join test with Filosofi, list cleanup commands.
"""

import os
import random
import zipfile
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT      = "/home/crusie/3. Code/Régions Inégales"
FIRMS_PANEL  = os.path.join(PROJECT, "firms_clean/firms_panel.csv")
FILO_PANEL   = os.path.join(PROJECT, "Base niveau administratif/filosofi_income_panel.csv")
ZIP_PATH     = os.path.join(PROJECT, "d'entreprises/DS_SIDE_CREA_DEP_REG_NAT_2024_CSV_FR.zip")
DATA_FILE    = "DS_SIDE_CREA_DEP_REG_NAT_2024_data.csv"

METRO_DEPS = (
    {f"{i:02d}" for i in range(1, 20)}
    | {f"{i:02d}" for i in range(21, 96)}
    | {"2A", "2B"}
)

task1_ok = False
task2_ok = False

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1, Spot check 8 random cells against raw data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("TASK 1, Spot check 8 random cells vs raw data")
print("=" * 65)

# Load panel
firms = pd.read_csv(FIRMS_PANEL, sep=";", dtype={"dep_code": str})
firms["year"] = firms["year"].astype(int)
print(f"\nFirms panel shape: {firms.shape}")
print(f"dep_code dtype: {firms['dep_code'].dtype}  sample: {firms['dep_code'].iloc[:3].tolist()}")

# Pick 8 random cells
random.seed(42)
all_combos = list(zip(firms["dep_code"], firms["year"]))
sampled = random.sample(all_combos, 8)
sampled_set = set(sampled)  # for fast lookup in streaming pass

print(f"\n8 sampled (dep_code, year) pairs:")
for dep, yr in sampled:
    row = firms[(firms["dep_code"] == dep) & (firms["year"] == yr)].iloc[0]
    print(f"  dep={dep}  year={yr}  total={int(row['total_firm_creations']):,}  "
          f"sarl={int(row['creations_sarl']):,}  sas={int(row['creations_sas']):,}")

# Re-compute totals from raw via ZIP streaming
# We want BURE + DEP + metro + the 8 specific (dep, year) combos.
# Accumulate sum of OBS_VALUE for ACTIVITY=_T and LEGAL_FORM=_T (no double count).
print(f"\nStreaming raw data from ZIP to recompute the 8 cells...")
raw_totals = {}   # (dep, year) → sum of OBS_VALUE

sampled_deps  = {dep for dep, _ in sampled}
sampled_years = {yr  for _, yr  in sampled}

total_rows = 0
with zipfile.ZipFile(ZIP_PATH) as zf:
    with zf.open(DATA_FILE) as fh:
        for i, chunk in enumerate(
            pd.read_csv(
                fh, sep=";", encoding="utf-8",
                chunksize=1_000_000,
                dtype={"GEO": str, "GEO_OBJECT": str, "ACTIVITY": str,
                       "LEGAL_FORM": str, "SIDE_MEASURE": str, "TIME_PERIOD": str},
            )
        ):
            total_rows += len(chunk)
            if i % 5 == 0:
                print(f"  {total_rows:,} rows scanned...")

            # Keep only what we need for the spot check
            c = chunk[
                (chunk["SIDE_MEASURE"] == "BURE") &
                (chunk["GEO_OBJECT"]   == "DEP")  &
                (chunk["ACTIVITY"]     == "_T")   &
                (chunk["LEGAL_FORM"]   == "_T")   &
                (chunk["GEO"].isin(sampled_deps)) &
                (chunk["TIME_PERIOD"].astype(int).isin(sampled_years))
            ].copy()

            if len(c) == 0:
                continue

            c["OBS_VALUE"]   = pd.to_numeric(c["OBS_VALUE"], errors="coerce").fillna(0)
            c["TIME_PERIOD"] = c["TIME_PERIOD"].astype(int)

            for _, row in c.iterrows():
                key = (row["GEO"], row["TIME_PERIOD"])
                if key in sampled_set:
                    raw_totals[key] = raw_totals.get(key, 0) + row["OBS_VALUE"]

print(f"\nTotal rows in raw file: {total_rows:,}")

# Build comparison table
print("\nSpot-check comparison table:")
header = f"{'dep':>6} | {'year':>4} | {'panel_total':>12} | {'raw_recomputed':>14} | match?"
print(header)
print("-" * len(header))

all_match = True
for dep, yr in sampled:
    row        = firms[(firms["dep_code"] == dep) & (firms["year"] == yr)].iloc[0]
    panel_val  = int(row["total_firm_creations"])
    raw_val    = int(raw_totals.get((dep, yr), -1))
    match      = "✓" if panel_val == raw_val else "✗ MISMATCH"
    if panel_val != raw_val:
        all_match = False
    print(f"{dep:>6} | {yr:>4} | {panel_val:>12,} | {raw_val:>14,} | {match}")

print()
if all_match:
    print("VERDICT: VERIFIED ✓ , all 8 cells match raw recomputed values")
    task1_ok = True
else:
    print("VERDICT: MISMATCH DETECTED, review details above")
    print("Stopping before cleanup. Fix the panel and re-run.")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 2, Test join with Filosofi
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("TASK 2, Join test with Filosofi panel")
print("=" * 65)

# Load Filosofi
filo = pd.read_csv(FILO_PANEL, sep=";", dtype={"dep_code": str})
filo["year"] = filo["year"].astype(int)
print(f"\nFilosofi panel shape: {filo.shape}")

# Dtype check before merge
print("\nKey column dtypes:")
print(f"  Filosofi  , dep_code: {filo['dep_code'].dtype}  year: {filo['year'].dtype}")
print(f"  Firms     , dep_code: {firms['dep_code'].dtype}  year: {firms['year'].dtype}")
print(f"  Filosofi   dep_code sample: {filo['dep_code'].head(3).tolist()}")
print(f"  Firms      dep_code sample: {firms['dep_code'].head(3).tolist()}")
print(f"  Filosofi   year sample:     {filo['year'].head(3).tolist()}")
print(f"  Firms      year sample:     {firms['year'].head(3).tolist()}")

dtype_ok = (
    str(filo["dep_code"].dtype) == str(firms["dep_code"].dtype) and
    str(filo["year"].dtype)     == str(firms["year"].dtype)
)
print(f"\nDtypes compatible: {'YES ✓' if dtype_ok else 'NO ✗, fix before merging'}")

# Inner merge
inner = pd.merge(filo, firms, on=["dep_code", "year"], how="inner")
print(f"\nInner merge shape: {inner.shape}  (expect (960, ...))")

inner_ok = inner.shape[0] == 960

if not inner_ok:
    # Diagnose: which keys are missing on each side
    filo_keys  = set(zip(filo["dep_code"],  filo["year"]))
    firms_keys = set(zip(firms["dep_code"], firms["year"]))
    in_filo_not_firms = filo_keys  - firms_keys
    in_firms_not_filo = firms_keys - filo_keys
    if in_filo_not_firms:
        print(f"  Keys in Filosofi but NOT in firms ({len(in_filo_not_firms)}):")
        for k in sorted(in_filo_not_firms)[:20]:
            print(f"    {k}")
    if in_firms_not_filo:
        print(f"  Keys in firms but NOT in Filosofi ({len(in_firms_not_filo)}):")
        for k in sorted(in_firms_not_filo)[:20]:
            print(f"    {k}")

# Outer merge
outer = pd.merge(filo, firms, on=["dep_code", "year"], how="outer")
print(f"Outer merge shape: {outer.shape}  (expect (960, ...))")

outer_ok = outer.shape[0] == 960

if outer_ok and inner_ok:
    print("\nJOIN VERDICT: PERFECT ALIGNMENT ✓ , inner=960, outer=960")
    task2_ok = True
else:
    print("\nJOIN VERDICT: ALIGNMENT ISSUE ✗")
    if not inner_ok:
        print(f"  Inner join has {inner.shape[0]} rows (expected 960)")
    if not outer_ok:
        print(f"  Outer join has {outer.shape[0]} rows (expected 960)")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 3, Cleanup commands (list only)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("TASK 3, Cleanup commands")
print("=" * 65)

zip_size_mb = os.path.getsize(ZIP_PATH) / 1e6
meta_size_kb = os.path.getsize(
    os.path.join(PROJECT, "firms_raw/DS_SIDE_CREA_DEP_REG_NAT_2024_metadata.csv")
) / 1e3

if task1_ok and task2_ok:
    print(f"""
Both verifications passed. Run these commands manually to free disk space:

  rm -rf "/home/crusie/3. Code/Régions Inégales/firms_raw/"
  rm -rf "/home/crusie/3. Code/Régions Inégales/d'entreprises/"

Disk space freed:
  firms_raw/  (metadata only)  ~{meta_size_kb:.0f} KB
  d'entreprises/  (ZIP)        ~{zip_size_mb:.0f} MB
  Total:                       ~{zip_size_mb:.0f} MB

Files that survive (and must NOT be deleted):
  Base niveau administratif/filosofi_income_panel.csv   ← Filosofi panel
  Base niveau administratif/column_coverage_*.csv       ← Filosofi coverage
  firms_clean/firms_panel.csv                           ← firm creations panel
  DATA_SOURCES.md                                       ← provenance log
""")
else:
    issues = []
    if not task1_ok:
        issues.append("Task 1 (spot check) found a mismatch in the firms panel")
    if not task2_ok:
        issues.append("Task 2 (join test) found misaligned keys between the two panels")
    print("\nDO NOT CLEAN UP, issues found:")
    for issue in issues:
        print(f"  ✗ {issue}")
    print("\nResolve the issues above, then re-run this script.")
