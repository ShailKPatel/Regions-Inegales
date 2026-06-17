"""
Master panel verification — doubled checks.
Layers 1, 2, 3 as specified, with 2× the original check count.
Run with:  python verify_master.py
"""
import pandas as pd
import numpy as np
import random

MASTER = "merged/france_panel_master.csv"

df = pd.read_csv(MASTER, sep=";", dtype={"dep_code": str})

FILOSOFI_COLS = [
    "n_households", "n_persons", "n_uc",
    "q1_dec", "q2_dec", "q3_dec", "d1_dec", "d9_dec",
    "gini_dec", "s80s20_dec", "d9_d1_dec",
    "q1_disp", "q2_disp", "q3_disp", "d1_disp", "d9_disp",
    "gini_disp", "s80s20_disp", "d9_d1_disp",
    "poverty_rate_disp", "poverty_rate_dec",
    "pct_wages", "pct_unemployment", "pct_capital_gains", "pct_pensions", "pct_other",
]
FIRMS_COLS = [
    "total_firm_creations", "creations_individual", "creations_sarl",
    "creations_sas", "creations_other_legal",
    "creations_sector_industry", "creations_sector_construction",
    "creations_sector_trade", "creations_sector_services",
]

print("=" * 70)
print("LAYER 1 — MERGE INTEGRITY (8 checks, 2× original)")
print("=" * 70)

# ── CHECK 1: Shape ──────────────────────────────────────────────────────────
n_rows, n_cols = df.shape
fail = n_rows != 960
print(f"\nCHECK 1 — Row count: {n_rows}  ({'PASS' if not fail else 'FAIL — expected 960'})")

# ── CHECK 2: Key uniqueness & leading zeros ─────────────────────────────────
n_unique_dep  = df["dep_code"].nunique()
n_unique_year = df["year"].nunique()
dup_keys      = df.duplicated(subset=["dep_code", "year"]).sum()
leading_zeros = (df["dep_code"].str.len() == 2).all()
year_range_ok = (df["year"].min() == 2012) and (df["year"].max() == 2021)
year_vals     = sorted(df["year"].unique())

print(f"\nCHECK 2 — Key integrity:")
print(f"  Unique dep_codes : {n_unique_dep}  {'PASS' if n_unique_dep==96 else 'FAIL expected 96'}")
print(f"  Unique years     : {n_unique_year} {year_vals}  {'PASS' if n_unique_year==10 else 'FAIL'}")
print(f"  Year range 2012-2021: {'PASS' if year_range_ok else 'FAIL'}")
print(f"  Duplicate (dep,year) pairs: {dup_keys}  {'PASS' if dup_keys==0 else 'FAIL'}")
print(f"  All dep_codes 2-char strings: {'PASS' if leading_zeros else 'FAIL — some have wrong length'}")

# ── CHECK 3: No all-null rows from outer-join mismatch ──────────────────────
all_filosofi_null = df[FILOSOFI_COLS].isnull().all(axis=1).sum()
all_firms_null    = df[FIRMS_COLS].isnull().all(axis=1).sum()
print(f"\nCHECK 3 — All-null side rows (outer-join ghost rows):")
print(f"  Rows where ALL Filosofi cols null: {all_filosofi_null}  {'PASS' if all_filosofi_null==0 else 'FAIL'}")
print(f"  Rows where ALL Firms cols null   : {all_firms_null}  {'PASS' if all_firms_null==0 else 'FAIL'}")

# ── CHECK 4: Cross-source correlation (income ↔ firms) ─────────────────────
valid = df[["q2_disp", "total_firm_creations"]].dropna()
corr = valid["q2_disp"].corr(valid["total_firm_creations"])
print(f"\nCHECK 4 — Income↔Firms correlation (should be clearly POSITIVE):")
print(f"  Pearson r(q2_disp, total_firm_creations) = {corr:.4f}  "
      f"{'PASS — positive' if corr > 0.3 else 'WARN — low/negative, check merge'}")

# ── CHECK 5: Population-normalised correlation ──────────────────────────────
# Per-capita firms vs income: should also be positive (but weaker, Paris is outlier)
valid2 = df[["q2_disp", "total_firm_creations", "n_persons"]].dropna()
valid2 = valid2[valid2["n_persons"] > 0].copy()
valid2["firms_per_1k"] = valid2["total_firm_creations"] / valid2["n_persons"] * 1000
corr2 = valid2["q2_disp"].corr(valid2["firms_per_1k"])
print(f"\nCHECK 5 — Income↔Firms-per-1000-persons correlation:")
print(f"  Pearson r = {corr2:.4f}  "
      f"('PASS — positive' if corr2 > 0.0 else 'WARN — negative')")

# ── CHECK 6: Spot-pairing check (7 departments) ────────────────────────────
print(f"\nCHECK 6 — Spot-pairings (7 departments, any year)")
SPOTS = [
    ("75", "Paris",           "high income, high firms"),
    ("93", "Seine-Saint-Denis","low income, high firms (populous)"),
    ("23", "Creuse",          "low income, low firms (small rural)"),
    ("48", "Lozère",          "low firms (smallest dept), modest income"),
    ("92", "Hauts-de-Seine",  "high income, high firms"),
    ("69", "Rhône",           "high income, high firms (Lyon)"),
    ("02", "Aisne",           "below-avg income, modest firms"),
]
for dep, name, expectation in SPOTS:
    row = df[(df["dep_code"] == dep) & (df["year"] == 2019)]
    if row.empty:
        row = df[df["dep_code"] == dep].sort_values("year").head(1)
    if row.empty:
        print(f"  {dep} {name}: NOT FOUND IN MASTER")
        continue
    r = row.iloc[0]
    print(f"  {dep} {r['dep_name']:25s} {int(r['year'])} | "
          f"q2_disp={r['q2_disp']:9.0f} €  poverty={r.get('poverty_rate_disp', float('nan')):5.1f}%  "
          f"firms={r['total_firm_creations']:8.0f}  [{expectation}]")

# ── CHECK 7: Temporal monotonicity of medians ──────────────────────────────
# National median income (average across depts) should trend upward over time
yearly_median = df.groupby("year")["q2_disp"].median()
diffs = yearly_median.diff().dropna()
n_up   = (diffs > 0).sum()
n_down = (diffs < 0).sum()
print(f"\nCHECK 7 — National median income trend (should mostly rise 2012-2021):")
for yr, val in yearly_median.items():
    print(f"  {yr}: {val:9.0f} €")
print(f"  Year-on-year: {n_up} up, {n_down} down  "
      f"{'PASS' if n_up >= 6 else 'WARN — unusual downturn pattern'}")

# ── CHECK 8: Internal additivity of firm sub-categories ───────────────────
# total_firm_creations should equal sum of creations_individual + sarl + sas + other
# (not necessarily exact — rounding/reclassification OK, but check magnitude)
sub_sum = df[["creations_individual","creations_sarl","creations_sas","creations_other_legal"]].sum(axis=1)
diff = (df["total_firm_creations"] - sub_sum).abs()
mean_diff_pct = (diff / df["total_firm_creations"].replace(0, np.nan)).mean() * 100
big_discrepancies = (diff > df["total_firm_creations"] * 0.05).sum()
print(f"\nCHECK 8 — Firm sub-category additivity (total ≈ sum of legal forms):")
print(f"  Mean absolute discrepancy: {mean_diff_pct:.2f}%  (warn if >5%)")
print(f"  Rows with >5% gap        : {big_discrepancies}")
if big_discrepancies == 0:
    print("  PASS — legal-form sub-totals add up cleanly")
else:
    print("  WARN — some rows have >5% discrepancy in sub-totals")

print("\n" + "=" * 70)
print("LAYER 1 SUMMARY")
print("=" * 70)
l1_pass = (
    n_rows == 960 and
    n_unique_dep == 96 and
    n_unique_year == 10 and
    dup_keys == 0 and
    leading_zeros and
    year_range_ok and
    all_filosofi_null == 0 and
    all_firms_null == 0 and
    corr > 0.3
)
print("MERGE INTEGRITY CONFIRMED — no row misalignment." if l1_pass else "ISSUE FOUND — see check failures above.")

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LAYER 2 — 30 RANDOM CELLS (web verification candidates)")
print("=" * 70)

random.seed(2024)
idx_sample = random.sample(range(len(df)), 30)
sample     = df.iloc[sorted(idx_sample)].copy()

print(f"\n{'dep':4s} {'name':25s} {'year':4s} {'q2_disp':>9s} {'pov_rate':>8s} {'firms':>8s}")
print("-" * 65)
cells_2021 = []
for _, row in sample.iterrows():
    dep  = row["dep_code"]
    name = row.get("dep_name", "?")
    yr   = int(row["year"])
    q2   = row["q2_disp"]
    pov  = row.get("poverty_rate_disp", float("nan"))
    firms= row["total_firm_creations"]
    print(f"{dep:4s} {str(name):25s} {yr:4d} {q2:9.0f} {pov:8.1f} {firms:8.0f}")
    if yr == 2021:
        cells_2021.append((dep, str(name), yr, q2, pov, firms))

print(f"\n2021 cells ready for web verification: {len(cells_2021)}")
for c in cells_2021:
    print(f"  dep={c[0]} name={c[1]} q2_disp={c[3]:.0f} pov={c[4]:.1f}% firms={c[5]:.0f}")

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LAYER 3 — NATIONAL FIRM TOTALS BY YEAR (master vs published)")
print("=" * 70)

nat_totals = df.groupby("year")["total_firm_creations"].sum().sort_index()
print(f"\n{'year':4s} | {'master_sum':>12s}")
print("-" * 22)
for yr, tot in nat_totals.items():
    print(f"{yr:4d} | {int(tot):>12,}")

print("\n[Now run web searches for published national totals — see below]")
print("Target queries:")
for yr in [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]:
    print(f"  {yr}: créations entreprises France {yr} nombre insee")
