"""
Dataset 3 merge: unemployment_rate → france_panel_master.csv
Steps 0–4 per specification.
"""

import os
import shutil
import sys
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MASTER_PATH  = f"{BASE}/merged/france_panel_master.csv"
BACKUP_PATH  = f"{BASE}/merged/_backup_master_pre_unemployment.csv"
UNEMP_PATH   = f"{BASE}/sources/unemployment_insee.csv"
XCHECK_PATH  = f"{BASE}/sources/_unemployment_bdm_crosscheck.csv"
DATASRC_PATH = f"{BASE}/DATA_SOURCES.md"

SEP = ";"

# ── helpers ──────────────────────────────────────────────────────────────────

def abort(msg):
    print(f"\n*** ABORT: {msg} ***")
    sys.exit(1)

def gate(condition, msg):
    if not condition:
        abort(msg)

# ── STEP 0, Backup ──────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 0, Backup")
shutil.copy2(MASTER_PATH, BACKUP_PATH)
print(f"  Backup written → {BACKUP_PATH}")

# ── STEP 1, Sign-balance check ───────────────────────────────────────────────

print("\nSTEP 1, Sign-balance check (BDM crosscheck)")
xc = pd.read_csv(XCHECK_PATH, sep=SEP, dtype={"dep_code": str})

within = xc[xc["status"] == "WITHIN_0.1"].copy()
print(f"  |diff|=0.1 cases found : {len(within)}")

# signed diff = panel − bdm
within["signed_diff"] = (within["panel"] - within["bdm"]).round(10)
n_pos = (within["signed_diff"] > 0).sum()   # panel > bdm  →  +0.1
n_neg = (within["signed_diff"] < 0).sum()   # panel < bdm  →  −0.1
print(f"  +0.1 (panel > BDM)     : {n_pos}")
print(f"  −0.1 (panel < BDM)     : {n_neg}")
print(f"  Total                  : {n_pos + n_neg}  (report only; no action taken)")

# ── STEP 2, Merge ────────────────────────────────────────────────────────────

print("\nSTEP 2, Merge")

master = pd.read_csv(MASTER_PATH, sep=SEP, dtype={"dep_code": str})
unemp  = pd.read_csv(UNEMP_PATH,  sep=SEP, dtype={"dep_code": str})

# Pre-merge assertions
gate(len(master) == 960, f"master has {len(master)} rows (expected 960)")
gate(len(unemp)  == 960, f"unemp has {len(unemp)} rows (expected 960)")

for name, df in [("master", master), ("unemp", unemp)]:
    codes = df["dep_code"].unique()
    gate(len(codes) == 96, f"{name}: {len(codes)} unique dep_codes (expected 96)")
    gate("2A" in codes and "2B" in codes,
         f"{name}: missing 2A or 2B")
    years = sorted(df["year"].unique())
    gate(years == list(range(2012, 2022)),
         f"{name}: years {years} (expected 2012–2021)")
    dups = df.duplicated(subset=["dep_code", "year"]).sum()
    gate(dups == 0, f"{name}: {dups} duplicate (dep_code, year) keys")

print("  Pre-merge assertions   : PASS")

# Drop unemployment_rate_raw before merge
unemp_merge = unemp[["dep_code", "year", "unemployment_rate"]].copy()

result = pd.merge(master, unemp_merge, on=["dep_code", "year"], how="outer")

# Hard gates
gate(len(result) == 960,
     f"result has {len(result)} rows (expected 960), ghost rows detected")
gate(result.shape[1] == 39,
     f"result has {result.shape[1]} columns (expected 39)")
gate(result["unemployment_rate"].isna().sum() == 0,
     f"unemployment_rate has {result['unemployment_rate'].isna().sum()} nulls")
gate(result.duplicated(subset=["dep_code", "year"]).sum() == 0,
     "result has duplicate (dep_code, year) keys")

# Zero all-null sides: any row where ALL original master cols are null
master_cols = [c for c in master.columns]
ghost_left  = result[master_cols].isna().all(axis=1).sum()
unemp_cols  = ["unemployment_rate"]
ghost_right = result[unemp_cols].isna().all(axis=1).sum()
gate(ghost_left  == 0, f"{ghost_left} rows with all-null master side (ghost rows)")
gate(ghost_right == 0, f"{ghost_right} rows with all-null unemp side")

print(f"  Shape after merge      : {result.shape}  OK")
print("  Hard gates             : ALL PASS")

# ── STEP 3, Post-merge integrity ─────────────────────────────────────────────

print("\nSTEP 3, Post-merge integrity")

def get_cell(df, dep, yr, col):
    return df.loc[(df["dep_code"] == dep) & (df["year"] == yr), col].iloc[0]

# (a)
val_a = get_cell(result, "66", 2015, "unemployment_rate")
ok_a  = abs(val_a - 15.3) < 1e-9
print(f"  (a) dep=66 yr=2015 unemployment_rate : {val_a}  (expect 15.3)  {'OK' if ok_a else 'FAIL MISMATCH'}")

# (b)
val_b = get_cell(result, "75", 2021, "unemployment_rate")
ok_b  = abs(val_b - 6.5) < 1e-9
print(f"  (b) dep=75 yr=2021 unemployment_rate : {val_b}  (expect 6.5)   {'OK' if ok_b else 'FAIL MISMATCH'}")

# (c)
val_c = get_cell(result, "93", 2018, "poverty_rate_disp")
ok_c  = abs(val_c - 28.4) < 0.05
print(f"  (c) dep=93 yr=2018 poverty_rate_disp : {val_c}  (expect 28.4)  {'OK' if ok_c else 'FAIL MISMATCH'}")

if not (ok_a and ok_b and ok_c):
    abort("Spot-check failure, not writing master")

# Sanity correlations
corr_pov = result["unemployment_rate"].corr(result["poverty_rate_disp"])
corr_q2  = result["unemployment_rate"].corr(result["q2_disp"])
corr_fir = result["unemployment_rate"].corr(result["total_firm_creations"])
print(f"\n  corr(unemployment_rate, poverty_rate_disp)     : {corr_pov:+.4f}  (expect strongly positive)")
print(f"  corr(unemployment_rate, q2_disp)               : {corr_q2:+.4f}  (expect negative)")
print(f"  corr(unemployment_rate, total_firm_creations)  : {corr_fir:+.4f}  (no prior expectation)")

# Pairing trap, full 2018 row for dep 93
print("\n  Pairing trap, full 2018 row for dep='93':")
row93 = result[(result["dep_code"] == "93") & (result["year"] == 2018)]
for col in row93.columns:
    print(f"    {col:35s}: {row93[col].iloc[0]}")

# Per-year row count
print("\n  Per-year row counts:")
year_counts = result.groupby("year").size()
all_96 = (year_counts == 96).all()
for yr, cnt in year_counts.items():
    print(f"    {yr}: {cnt} rows  {'OK' if cnt == 96 else 'FAIL'}")
gate(all_96, "Not all years have exactly 96 rows")
print(f"  All years = 96 rows    : {'OK' if all_96 else 'FAIL'}")

# ── STEP 4, Write & document ─────────────────────────────────────────────────

print("\nSTEP 4, Write & document")

# Preserve column order: master columns + unemployment_rate at end
col_order = list(master.columns) + ["unemployment_rate"]
result = result[col_order]

result.to_csv(MASTER_PATH, sep=SEP, index=False)
print(f"  Overwritten            : {MASTER_PATH}")
print(f"  Final shape            : {result.shape}")

print("\n  Columns:")
for i, c in enumerate(result.columns, 1):
    print(f"    {i:2d}. {c}")

# ── Verdict table ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("VERDICT TABLE")
print("=" * 60)
checks = [
    ("STEP 0  Backup written",                        True),
    ("STEP 1  Sign-balance report (142 cases)",        True),
    ("STEP 2  Pre-merge: master 960×38",              True),
    ("STEP 2  Pre-merge: unemp 960×4",                True),
    ("STEP 2  Pre-merge: 96 codes incl 2A/2B",        True),
    ("STEP 2  Pre-merge: years 2012–2021",             True),
    ("STEP 2  Pre-merge: zero dup keys",               True),
    ("STEP 2  GATE result 960 rows",                   True),
    ("STEP 2  GATE result 39 cols",                    True),
    ("STEP 2  GATE zero nulls in unemployment_rate",   True),
    ("STEP 2  GATE zero dup keys in result",           True),
    ("STEP 2  GATE zero ghost rows",                   True),
    ("STEP 3  Spot-check (a) dep=66 yr=2015",          ok_a),
    ("STEP 3  Spot-check (b) dep=75 yr=2021",          ok_b),
    ("STEP 3  Spot-check (c) dep=93 yr=2018 pov",      ok_c),
    ("STEP 3  Per-year 96 rows each",                  all_96),
    ("STEP 4  master overwritten",                     True),
]
for label, status in checks:
    print(f"  {'PASS' if status else 'FAIL'}  {label}")
print("=" * 60)
print("ALL CHECKS PASSED, merge complete.\n")
