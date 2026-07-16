"""
DATASET 4: INSEE population panel extraction + DREES density cross-verification.
Tasks A and B per project spec.
"""

import csv
import openpyxl
import pandas as pd

YEARS = list(range(2012, 2022))   # 2012–2021 inclusive
POP_FILE = "population_raw/estim-pop-dep-sexe-gca-1975-2026.xlsx"
DENSITY_FILE = "sources/doctor_density_drees.csv"
POP_OUT = "sources/population_insee.csv"
CHECK_OUT = "sources/_doctor_density_pop_crosscheck.csv"

# ---------------------------------------------------------------------------
# TASK A, extract population panel
# ---------------------------------------------------------------------------

def is_metro_dep(code):
    """True for string codes 01-95, 2A, 2B (metro France only)."""
    if not isinstance(code, str):
        return False
    c = code.strip()
    if c in ("2A", "2B"):
        return True
    try:
        n = int(c)
        return 1 <= n <= 95
    except ValueError:
        return False


print("=== TASK A: extracting population panel ===")
wb = openpyxl.load_workbook(POP_FILE, read_only=True, data_only=True)

records = []
for year in YEARS:
    ws = wb[str(year)]
    for row in ws.iter_rows(values_only=True):
        code = row[0]
        pop = row[7]   # Ensemble > Total column (H)
        if is_metro_dep(code) and pop is not None:
            records.append({
                "dep_code": code.strip(),
                "year": year,
                "pop_jan1": int(pop),
            })

wb.close()

df_pop = pd.DataFrame(records)
print(f"Rows extracted: {len(df_pop)}")
assert len(df_pop) == 960, f"Expected 960, got {len(df_pop)}"
assert df_pop.isnull().sum().sum() == 0, "Nulls found"
assert df_pop.duplicated(["dep_code", "year"]).sum() == 0, "Duplicate keys found"

# Plausibility checks
lozere = df_pop[df_pop.dep_code == "48"]
nord = df_pop[df_pop.dep_code == "59"]
paris = df_pop[df_pop.dep_code == "75"].sort_values("year")

print(f"\nLozère (48) pop range: {lozere.pop_jan1.min():,} – {lozere.pop_jan1.max():,}")
print(f"Nord (59) pop range:   {nord.pop_jan1.min():,} – {nord.pop_jan1.max():,}")
print("Paris (75) by year:")
for _, r in paris.iterrows():
    print(f"  {r.year}: {r.pop_jan1:,}")

# Save, dep_code quoted, semicolon separator
with open(POP_OUT, "w", newline="", encoding="utf-8") as f:
    f.write('"dep_code";year;pop_jan1\n')
    for _, r in df_pop.iterrows():
        f.write(f'"{r.dep_code}";{r.year};{int(r.pop_jan1)}\n')

print(f"\nSaved: {POP_OUT}  ({len(df_pop)} rows × 3 cols)")

# ---------------------------------------------------------------------------
# TASK B, density cross-verification
# ---------------------------------------------------------------------------

print("\n=== TASK B: density cross-verification ===")

df_dens = pd.read_csv(DENSITY_FILE, sep=";", dtype={"dep_code": str})
print(f"doctor_density_drees: {df_dens.shape}, cols: {list(df_dens.columns)}")

# Normalise dep_code in density file (leading zero guard)
df_dens["dep_code"] = df_dens["dep_code"].str.strip().str.zfill(2)
df_pop["dep_code"] = df_pop["dep_code"].str.strip().str.zfill(2)

merged = df_dens.merge(df_pop, on=["dep_code", "year"], how="inner")
print(f"Merged rows: {len(merged)}  (expected 960)")
assert len(merged) == 960, f"Merge yielded {len(merged)} rows"

# Recompute density
merged["density_check"] = merged["n_doctors"] / merged["pop_jan1"] * 100_000

# Relative difference
merged["rel_diff"] = (merged["density_check"] - merged["doctor_density_per_100k"]) / merged["doctor_density_per_100k"]

abs_diff = merged["rel_diff"].abs()

n_05  = (abs_diff <= 0.005).sum()
n_1   = (abs_diff <= 0.01).sum()
n_gt1 = (abs_diff >  0.01).sum()

print(f"\nRelative-diff distribution (960 cells):")
print(f"  within ±0.5% : {n_05}  ({n_05/9.6:.1f}%)")
print(f"  within ±1.0% : {n_1}  ({n_1/9.6:.1f}%)")
print(f"  beyond ±1.0% : {n_gt1}  ({n_gt1/9.6:.1f}%)")

beyond = merged[abs_diff > 0.01].copy()
if len(beyond) > 0:
    print(f"\nCells beyond ±1% ({len(beyond)}):")
    print(f"{'dep_code':>8} {'year':>4} {'drees':>10} {'recomp':>10} {'rel_diff%':>10} {'implied_pop':>12}")
    for _, r in beyond.sort_values("rel_diff", ascending=False).iterrows():
        implied = r["n_doctors"] / r["doctor_density_per_100k"] * 100_000
        print(f"{r.dep_code:>8} {r.year:>4} {r.doctor_density_per_100k:>10.3f} {r.density_check:>10.3f} {r.rel_diff*100:>9.2f}% {implied:>12,.0f}")
    # Cluster analysis
    print("\nBy year:")
    print(beyond.groupby("year").size().to_string())
    print("\nBy dep_code (top 10):")
    print(beyond.groupby("dep_code").size().nlargest(10).to_string())
else:
    print("\nNo cells beyond ±1%, perfect agreement.")

# PASS/FAIL
n_gt2 = (abs_diff > 0.02).sum()
print(f"\nBeyond ±2%: {n_gt2}")

passed = (n_1 / 960 >= 0.99) and (n_gt2 == 0)
verdict = "PASS" if passed else "FAIL"
print(f"\nVERDICT: {verdict}  ({n_1}/960 within ±1%;  {n_gt2} beyond ±2%)")

# Summary stats on relative diff
print(f"\nRel-diff stats:")
print(merged["rel_diff"].describe().apply(lambda x: f"{x*100:.4f}%"))

# Save crosscheck file
merged["rel_diff_pct"] = merged["rel_diff"] * 100
out_cols = ["dep_code", "year", "n_doctors", "doctor_density_per_100k", "pop_jan1", "density_check", "rel_diff_pct"]
merged[out_cols].to_csv(CHECK_OUT, sep=";", index=False, float_format="%.6f")
print(f"\nSaved: {CHECK_OUT}  ({len(merged)} rows)")

# Final verdict table
print("\n" + "="*60)
print("VERDICT TABLE")
print("="*60)
print(f"Population panel:   960 rows, 0 nulls, 0 dup keys     OK")
print(f"Lozère range:       {lozere.pop_jan1.min():,}–{lozere.pop_jan1.max():,}  (expect 75–80k min)")
print(f"Nord max:           {nord.pop_jan1.max():,}  (expect ~2.6M)")
paris_trend = "DECLINING" if paris.sort_values("year").pop_jan1.iloc[-1] < paris.sort_values("year").pop_jan1.iloc[0] else "NOT declining"
print(f"Paris trend:        {paris_trend}")
print(f"Cells within ±1%:  {n_1}/960 = {n_1/9.6:.1f}%")
print(f"Cells beyond ±2%:  {n_gt2}")
print(f"PASS criterion:     ≥99% within ±1% AND 0 beyond ±2%")
print(f"VERDICT:            {verdict}")
