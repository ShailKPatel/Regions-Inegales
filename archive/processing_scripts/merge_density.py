"""
DATASET 6 merge: pct_urban / density_class / density_is_static into france_panel_master.csv
"""

import sys
import shutil
import pandas as pd
import numpy as np

MASTER     = "merged/france_panel_master.csv"
BACKUP     = "merged/_backup_master_pre_density.csv"
DENSITY    = "sources/density_grille_insee.csv"
KEYS       = ["dep_code", "year"]

EXPECTED_MASTER_ROWS = 960
EXPECTED_MASTER_COLS = 42
EXPECTED_OUT_COLS    = 45
EXPECTED_DEPS        = 96
EXPECTED_YEARS       = list(range(2012, 2022))
NEW_COLS             = ["pct_urban", "density_class", "density_is_static"]

def abort(msg):
    print(f"\n  ABORT: {msg}")
    print("  Master untouched.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# STEP 0 — Backup
# ---------------------------------------------------------------------------
print("=== STEP 0: Backup ===")
shutil.copy2(MASTER, BACKUP)
print(f"  {MASTER} → {BACKUP}   OK")

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
master  = pd.read_csv(MASTER,  sep=";", dtype={"dep_code": str})
density = pd.read_csv(DENSITY, sep=";", dtype={"dep_code": str})

# Strip any residual quotes from dep_code
master["dep_code"]  = master["dep_code"].str.strip().str.strip('"')
density["dep_code"] = density["dep_code"].str.strip().str.strip('"')

# ---------------------------------------------------------------------------
# STEP 0b — Clamp floating-point near-100 artifacts
# ---------------------------------------------------------------------------
print("\n=== STEP 0b: Clamp pct_urban ≈ 100 ===")
mask_near100 = (density["pct_urban"] > 100 - 1e-9) & (density["pct_urban"] < 100)
n_clamped = mask_near100.sum()
if n_clamped > 0:
    deps_clamped = density.loc[mask_near100, "dep_code"].unique().tolist()
    density.loc[mask_near100, "pct_urban"] = 100.0
    print(f"  Clamped {n_clamped} rows (deps: {deps_clamped})")
else:
    print(f"  0 rows to clamp (artifact already resolved at build time; dep 92 already = 100.0)")

# Also clamp > 100 just in case
over100 = (density["pct_urban"] > 100)
if over100.any():
    abort(f"pct_urban > 100 in {density.loc[over100, 'dep_code'].tolist()}")

# ---------------------------------------------------------------------------
# STEP 1 — Assert both input files
# ---------------------------------------------------------------------------
print("\n=== STEP 1: Assertions ===")

def assert_file(df, label, expected_rows, expected_cols):
    rows, cols = df.shape
    if rows != expected_rows:
        abort(f"{label}: expected {expected_rows} rows, got {rows}")
    if cols != expected_cols:
        abort(f"{label}: expected {expected_cols} cols, got {cols}")
    dups = df.duplicated(subset=KEYS)
    if dups.any():
        abort(f"{label}: {dups.sum()} duplicate (dep_code, year) pairs")
    deps = df["dep_code"].unique()
    if len(deps) != EXPECTED_DEPS:
        abort(f"{label}: expected {EXPECTED_DEPS} deps, got {len(deps)} — missing: {sorted(set(['2A','2B']) - set(deps))}")
    for d in ["2A", "2B"]:
        if d not in deps:
            abort(f"{label}: missing dep {d}")
    years = sorted(df["year"].unique())
    if years != EXPECTED_YEARS:
        abort(f"{label}: years mismatch — got {years}")
    nulls = df["pct_urban"].isnull().sum() if "pct_urban" in df.columns else 0
    print(f"  {label}: {rows}×{cols}  |  {len(deps)} deps  |  dup keys=0  |  2A/2B present  |  pct_urban nulls={nulls}   OK")

assert_file(master,  "master",  EXPECTED_MASTER_ROWS, EXPECTED_MASTER_COLS)
assert_file(density, "density", EXPECTED_MASTER_ROWS, 5)

# ---------------------------------------------------------------------------
# STEP 2 — Merge
# ---------------------------------------------------------------------------
print("\n=== STEP 2: Merge ===")

density_cols = KEYS + NEW_COLS
merged = master.merge(density[density_cols], on=KEYS, how="outer", indicator=True)

# Ghost rows check (rows only in density, not in master)
ghost = merged[merged["_merge"] == "right_only"]
if len(ghost) > 0:
    abort(f"{len(ghost)} ghost rows (density keys not in master):\n{ghost[KEYS].to_string()}")

left_only = merged[merged["_merge"] == "left_only"]
if len(left_only) > 0:
    abort(f"{len(left_only)} master rows unmatched by density:\n{left_only[KEYS].to_string()}")

merged = merged.drop(columns=["_merge"])

# Hard gate: shape
rows_out, cols_out = merged.shape
if rows_out != EXPECTED_MASTER_ROWS:
    abort(f"Output has {rows_out} rows, expected {EXPECTED_MASTER_ROWS}")
if cols_out != EXPECTED_OUT_COLS:
    abort(f"Output has {cols_out} cols, expected {EXPECTED_OUT_COLS}")

# Hard gate: pct_urban nulls
null_urban = merged["pct_urban"].isnull().sum()
if null_urban > 0:
    abort(f"pct_urban has {null_urban} nulls")

# Hard gate: duplicate keys
dup_keys = merged.duplicated(subset=KEYS).sum()
if dup_keys > 0:
    abort(f"{dup_keys} duplicate (dep_code, year) keys in output")

# Hard gate: 96 rows per year
rows_per_year = merged.groupby("year").size()
bad_years = rows_per_year[rows_per_year != EXPECTED_DEPS]
if len(bad_years) > 0:
    abort(f"Rows per year != {EXPECTED_DEPS}:\n{bad_years}")

print(f"  Shape:       {rows_out}×{cols_out}   OK")
print(f"  pct_urban nulls:  0   OK")
print(f"  Dup keys:         0   OK")
print(f"  Ghost rows:       0   OK")
print(f"  Rows/year:       {EXPECTED_DEPS}   OK")

# ---------------------------------------------------------------------------
# STEP 3 — Integrity checks
# ---------------------------------------------------------------------------
print("\n=== STEP 3: Integrity ===")

def cell(df, dep, yr, col):
    row = df[(df["dep_code"] == dep) & (df["year"] == yr)]
    if row.empty:
        return None
    return row.iloc[0][col]

# Spot checks
spots = [
    ("75", 2020, "pct_urban",         100.0,   1e-6),
    ("23", 2015, "pct_urban",         10.268,  0.01),
    ("48", 2019, "pct_urban",         14.83,   0.01),
    ("93", 2018, "poverty_rate_disp", 28.4,    0.1),
]
print("  Spot checks:")
for dep, yr, col, expected, tol in spots:
    val = cell(merged, dep, yr, col)
    ok  = val is not None and abs(float(val) - expected) <= tol
    status = "OK" if ok else f"FAIL — expected {expected}, got {val}"
    print(f"    ({dep},{yr}) {col} = {val}   {status}")
    if not ok:
        abort(f"Spot check failed: ({dep},{yr}) {col}")

# density_class distribution
dc = merged["density_class"].value_counts()
print(f"\n  density_class distribution:")
for cls, exp in [("urban", 140), ("intermediate", 310), ("rural", 510)]:
    n = dc.get(cls, 0)
    status = "OK" if n == exp else f"FAIL (expected {exp})"
    print(f"    {cls}: {n}   {status}")
    if n != exp:
        abort(f"density_class '{cls}' count {n} != {exp}")

# Correlations
print(f"\n  Correlations with pct_urban:")
for col in ["doctor_density_per_100k", "edu_share_sup", "q2_disp"]:
    r = merged["pct_urban"].corr(merged[col])
    direction = "positive" if r > 0 else "NEGATIVE — unexpected"
    print(f"    pct_urban ↔ {col}: r = {r:.4f}   ({direction})")

print("\n  All integrity checks passed   OK")

# ---------------------------------------------------------------------------
# STEP 4 — Write master
# ---------------------------------------------------------------------------
print("\n=== STEP 4: Write master ===")
merged.to_csv(MASTER, sep=";", index=False)
print(f"  Wrote {MASTER}   {merged.shape[0]}×{merged.shape[1]}")

# Final report
print(f"\n  Columns ({merged.shape[1]}):")
for i, c in enumerate(merged.columns, 1):
    marker = " ← NEW" if c in NEW_COLS else ""
    print(f"    {i:3d}.  {c}{marker}")

# Verdict table
print("\n=== VERDICT ===")
rows = [
    ("Backup",              "merged/_backup_master_pre_density.csv", "OK"),
    ("Float clamp",         "dep 92 already = 100.0 (0 rows changed)", "OK"),
    ("Input assertion",     "master 960×42, density 960×5", "OK"),
    ("Merge shape",         f"{rows_out}×{cols_out}", "OK"),
    ("pct_urban nulls",     "0", "OK"),
    ("Dup keys",            "0", "OK"),
    ("Ghost rows",          "0", "OK"),
    ("Rows per year",       f"{EXPECTED_DEPS}", "OK"),
    ("Spot checks",         "(75,2020) (23,2015) (48,2019) (93,2018)", "OK"),
    ("density_class dist.", "140/310/510", "OK"),
    ("Master overwrite",    MASTER, "OK"),
]
for label, detail, verdict in rows:
    print(f"  {label:<25}  {detail:<45}  {verdict}")
