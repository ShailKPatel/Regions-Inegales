"""
DATASET 4 merge: doctor_density_per_100k into france_panel_master.csv
Steps 0-4 per project spec. Aborts before touching master if any gate fails.
"""

import shutil
import sys
import pandas as pd

MASTER     = "merged/france_panel_master.csv"
BACKUP     = "merged/_backup_master_pre_doctors.csv"
DENSITY    = "sources/doctor_density_drees.csv"
METRO_DEPS = 96
YEARS      = list(range(2012, 2022))
KEYS       = ["dep_code", "year"]

def abort(msg):
    print(f"\nABORT: {msg}")
    print("Master untouched.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# STEP 0, backup
# ---------------------------------------------------------------------------
print("STEP 0, backup")
shutil.copy2(MASTER, BACKUP)
print(f"  Backed up → {BACKUP}")

# ---------------------------------------------------------------------------
# STEP 1, re-assert both inputs
# ---------------------------------------------------------------------------
print("\nSTEP 1, re-assert inputs")

master = pd.read_csv(MASTER, sep=";", dtype={"dep_code": str})
density = pd.read_csv(DENSITY, sep=";", dtype={"dep_code": str})

for label, df in [("master", master), ("density", density)]:
    if len(df) != 960:
        abort(f"{label}: expected 960 rows, got {len(df)}")
    if df.duplicated(KEYS).sum():
        abort(f"{label}: duplicate (dep_code, year) keys found")
    deps = set(df.dep_code.unique())
    if len(deps) != METRO_DEPS:
        abort(f"{label}: expected {METRO_DEPS} unique deps, got {len(deps)}")
    if "2A" not in deps or "2B" not in deps:
        abort(f"{label}: 2A/2B not found in dep_code")
    years = sorted(df.year.unique())
    if years != YEARS:
        abort(f"{label}: expected years {YEARS}, got {years}")
    print(f"  {label}: 960 rows, 96 deps (2A/2B ok), years 2012–2021, no dup keys  OK")

if master.shape[1] != 39:
    abort(f"Master expected 39 cols, got {master.shape[1]}")
print("  Master column count 39  OK")

# ---------------------------------------------------------------------------
# STEP 2, outer merge, single new column
# ---------------------------------------------------------------------------
print("\nSTEP 2, merge")

density_col = density[KEYS + ["doctor_density_per_100k"]]

merged = master.merge(density_col, on=KEYS, how="outer")

# HARD GATES
if len(merged) != 960:
    abort(f"Outer merge yielded {len(merged)} rows (expected 960), key mismatch")
if merged.duplicated(KEYS).sum():
    abort("Duplicate keys after merge")
if merged["doctor_density_per_100k"].isnull().sum():
    abort(f"Nulls in doctor_density_per_100k: {merged['doctor_density_per_100k'].isnull().sum()}")
if merged.shape[1] != 40:
    abort(f"Expected 960×40, got {merged.shape}")
# ghost rows: any row where ALL original master cols are null
master_cols = [c for c in master.columns]
ghost = merged[merged[master_cols].isnull().all(axis=1)]
if len(ghost):
    abort(f"{len(ghost)} ghost rows (unmatched density keys)")

print(f"  Shape: {merged.shape}  OK")
print(f"  Nulls in new col: 0  OK")
print(f"  Dup keys: 0  OK")
print(f"  Ghost rows: 0  OK")

# ---------------------------------------------------------------------------
# STEP 3, integrity
# ---------------------------------------------------------------------------
print("\nSTEP 3, integrity checks")

def cell(df, dep, yr, col):
    return df.loc[(df.dep_code == dep) & (df.year == yr), col].iloc[0]

# Spot cells
checks = [
    ("75", 2021, "doctor_density_per_100k", 894.14),
    ("27", 2019, "doctor_density_per_100k", 171.31),
    ("93", 2018, "poverty_rate_disp",        28.4),
]
for dep, yr, col, expected in checks:
    val = cell(merged, dep, yr, col)
    ok = abs(val - expected) < 0.01
    status = "OK" if ok else f"FAIL (expected {expected})"
    print(f"  ({dep},{yr}) {col} = {val}  {status}")
    if not ok:
        abort(f"Spot check failed: ({dep},{yr}) {col} = {val}, expected {expected}")

# Correlations (report only)
print("\n  Correlations with doctor_density_per_100k:")
for col in ["q2_disp", "unemployment_rate", "poverty_rate_disp"]:
    r = merged["doctor_density_per_100k"].corr(merged[col])
    print(f"    density ↔ {col}: r = {r:.4f}")

# Pairing trap
print("\n  Pairing trap, (05, 2021) full row:")
row05 = merged[(merged.dep_code == "05") & (merged.year == 2021)].iloc[0]
print(f"    dep={row05.dep_code}  year={row05.year}  dep_name={row05.dep_name}")
print(f"    doctor_density_per_100k = {row05.doctor_density_per_100k}")
print(f"    q2_disp                 = {row05.q2_disp}")
print(f"    unemployment_rate       = {row05.unemployment_rate}")
print(f"    poverty_rate_disp       = {row05.poverty_rate_disp}")

print("\n  Pairing trap, (27, 2019) Eure low-density profile:")
row27 = merged[(merged.dep_code == "27") & (merged.year == 2019)].iloc[0]
print(f"    dep={row27.dep_code}  year={row27.year}  dep_name={row27.dep_name}")
print(f"    doctor_density_per_100k = {row27.doctor_density_per_100k}")
print(f"    q2_disp                 = {row27.q2_disp}")
print(f"    unemployment_rate       = {row27.unemployment_rate}")
print(f"    poverty_rate_disp       = {row27.poverty_rate_disp}")

# 96 per year check
year_counts = merged.groupby("year").size()
if (year_counts != 96).any():
    abort(f"Not exactly 96 rows per year:\n{year_counts}")
print(f"\n  96 rows × 10 years: {dict(year_counts)}  OK")

# ---------------------------------------------------------------------------
# STEP 4, overwrite master
# ---------------------------------------------------------------------------
print("\nSTEP 4, overwriting master")

# dep_code must stay quoted (str, leading zeros)
merged.to_csv(MASTER, sep=";", index=False)

# Verify round-trip
verify = pd.read_csv(MASTER, sep=";", dtype={"dep_code": str})
assert verify.shape == (960, 40), f"Round-trip shape wrong: {verify.shape}"
assert verify["doctor_density_per_100k"].isnull().sum() == 0
assert verify.dep_code.str.match(r"^\d{2}$|^2[AB]$").all()
print(f"  Written and verified: {MASTER}  {verify.shape}  OK")

# ---------------------------------------------------------------------------
# VERDICT
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("VERDICT TABLE")
print("="*60)
print(f"Backup:            {BACKUP}                        OK")
print(f"Input assertions:  master 960×39, density 960×4   OK")
print(f"Merge shape:       960×40                         OK")
print(f"New col nulls:     0                              OK")
print(f"Ghost rows:        0                              OK")
print(f"Dup keys:          0                              OK")
print(f"Spot (75,2021):    {cell(verify,'75',2021,'doctor_density_per_100k')}                    OK")
print(f"Spot (27,2019):    {cell(verify,'27',2019,'doctor_density_per_100k')}                  OK")
print(f"Spot (93,2018)     poverty_rate_disp = {cell(verify,'93',2018,'poverty_rate_disp')}           OK")
print(f"96×10 balance:     confirmed                      OK")
print(f"Master:            overwritten 960×40             DONE")
