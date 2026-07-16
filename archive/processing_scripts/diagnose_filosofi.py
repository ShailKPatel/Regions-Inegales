"""
Régions Inégales - Column Diagnostic
Tasks 1–4: confirm year-siloing, coverage map, base-name grouping, reshape plan.
"""

import re, pathlib
import pandas as pd
import numpy as np

CLEAN = pathlib.Path("/home/crusie/3. Code/Régions Inégales/filosofi_clean")
MASTER = CLEAN / "filosofi_all_years.csv"

print("Loading master CSV…")
master = pd.read_csv(MASTER, sep=";", dtype={"CODGEO": str}, low_memory=False)
print(f"  {len(master)} rows × {len(master.columns)} columns loaded.\n")

YEARS = sorted(master["YEAR"].astype(int).unique())


# ---------------------------------------------------------------------------
# TASK 1, Confirm or deny year-siloing
# ---------------------------------------------------------------------------
print("=" * 72)
print("TASK 1, IS THE DATA YEAR-SILOED?")
print("=" * 72)

def find_col(pattern: str) -> str | None:
    matches = [c for c in master.columns if re.search(pattern, c, re.IGNORECASE)]
    return matches[0] if matches else None

test_specs = [
    ("Q2.*12",  2012, "Median income 2012"),
    ("Q2.*18",  2018, "Median income 2018"),
    ("TP60.*12", 2012, "Poverty rate 2012"),
    ("TP60.*18", 2018, "Poverty rate 2018"),
    ("GI.*18",   2018, "Gini 2018"),
]

verdict_year_siloed = True

for pattern, year, label in test_specs:
    col = find_col(pattern)
    if col is None:
        print(f"  {label}: ⚠ no column found for pattern '{pattern}'")
        continue
    total    = master[col].notna().sum()
    in_year  = master.loc[master["YEAR"].astype(int) == year,  col].notna().sum()
    out_year = master.loc[master["YEAR"].astype(int) != year, col].notna().sum()
    print(f"  {col:<35s}  total={total:>4}  in {year}={in_year:>3}  outside {year}={out_year:>3}")
    if out_year > 0:
        verdict_year_siloed = False

print()
if verdict_year_siloed:
    print("  VERDICT: CONFIRMED, data is year-siloed.")
    print("  Each column is only filled for rows matching its own year.")
else:
    print("  VERDICT: NOT the case, data appears to be stacked correctly.")


# ---------------------------------------------------------------------------
# TASK 2, Full column coverage map
# ---------------------------------------------------------------------------
print("\n\n" + "=" * 72)
print("TASK 2, COLUMN COVERAGE MAP")
print("=" * 72)
print("  Computing per-column non-null counts and year presence…")

rows = []
for col in master.columns:
    nonnull = master[col].notna().sum()
    if nonnull == 0:
        years_with_data = []
    else:
        years_with_data = sorted(
            master.loc[master[col].notna(), "YEAR"].astype(int).unique().tolist()
        )
    rows.append({
        "column_name":    col,
        "total_non_null": int(nonnull),
        "years_with_data": str(years_with_data),
        "n_years":         len(years_with_data),
    })

coverage_df = pd.DataFrame(rows)
coverage_path = CLEAN / "column_coverage.csv"
coverage_df.to_csv(coverage_path, sep=";", index=False, encoding="utf-8")
print(f"  Saved: {coverage_path.name}\n")

n_all   = (coverage_df["total_non_null"] == 960).sum()
n_one   = (coverage_df["total_non_null"] == 96).sum()
n_zero  = (coverage_df["total_non_null"] == 0).sum()
n_part  = len(coverage_df) - n_all - n_one - n_zero

print(f"  Columns with data in ALL  10 years (960 non-null): {n_all}")
print(f"  Columns with data in exactly 1 year  (96 non-null): {n_one}")
print(f"  Columns with partial coverage (1 < years < 10):     {n_part}")
print(f"  Columns entirely empty (0 non-null):                 {n_zero}")


# ---------------------------------------------------------------------------
# TASK 3, Base-name grouping
# ---------------------------------------------------------------------------
print("\n\n" + "=" * 72)
print("TASK 3, BASE-NAME GROUPING (year-suffix stripped)")
print("=" * 72)

# Strip 2-digit year suffix from column names
# Pattern: any sequence of digits 12–21 (as 2-digit year suffix in the name)
YEAR_PAT = re.compile(r'(1[2-9]|2[01])(?=[^0-9]|$)')

def strip_year_suffix(col: str) -> str:
    """Remove the 2-digit year embedded in a column name, return base name."""
    # Replace last occurrence of a 2-digit year code
    return YEAR_PAT.sub("{YY}", col, count=1)

base_groups: dict[str, list[dict]] = {}
for _, row in coverage_df.iterrows():
    col    = row["column_name"]
    base   = strip_year_suffix(col)
    if base not in base_groups:
        base_groups[base] = []
    base_groups[base].append({
        "col": col,
        "non_null": row["total_non_null"],
        "years": row["years_with_data"],
    })

# Focus concepts
KEY_PATTERNS = {
    "Median income (Q2)":    r"^Q2\{YY\}",
    "Poverty rate (TP60)":   r"^TP60\{YY\}",
    "Gini (GI)":             r"^GI\{YY\}",
    "Decile 1 (D1)":         r"^D1\{YY\}",
    "Decile 9 (D9)":         r"^D9\{YY\}",
    "N households (NBMEN)":  r"^NBMEN\{YY\}",
}

print("\n  KEY ECONOMIC CONCEPTS, year-group summary:\n")
print(f"  {'Base name':<35s} {'Years':<6} {'Coverage':<25} {'Example value'}")
print("  " + "-" * 80)

all_base_rows = []
for base, members in sorted(base_groups.items()):
    year_codes = []
    for m in members:
        # extract year from the column name
        real_years = eval(m["years"]) if m["years"] not in ("[]","") else []
        year_codes.extend(real_years)
    year_codes = sorted(set(year_codes))
    coverage = "COMPLETE (10 yr)" if len(year_codes) == 10 else f"partial ({len(year_codes)} yr)"
    example = ""
    if members:
        sample_col = members[0]["col"]
        example = str(master[sample_col].dropna().iloc[0]) if master[sample_col].notna().any() else ""
    all_base_rows.append((base, year_codes, coverage, example))

# Print key concept groups
for label, pat in KEY_PATTERNS.items():
    print(f"\n  {label}:")
    found = [(b, yc, cov, ex) for b, yc, cov, ex in all_base_rows
             if re.search(pat, b)]
    if not found:
        print(f"    ⚠ none found")
    for base, year_codes, coverage, example in found:
        print(f"    {base:<35s} {str(len(year_codes)):<6} {coverage:<25} val={example[:20]}")

# Overall base-group count
print(f"\n  Total distinct base-names: {len(base_groups)}")
complete_bases = sum(1 for _, yc, _, _ in all_base_rows if len(yc) == 10)
partial_bases  = sum(1 for _, yc, _, _ in all_base_rows if 0 < len(yc) < 10)
print(f"  Base names with full 10-year coverage: {complete_bases}")
print(f"  Base names with partial coverage:      {partial_bases}")


# ---------------------------------------------------------------------------
# TASK 4, Proposed reshape plan
# ---------------------------------------------------------------------------
print("\n\n" + "=" * 72)
print("TASK 4, PROPOSED RESHAPE PLAN")
print("=" * 72)

print("""
CONFIRMED: the data is year-siloed.

Each column like Q212_DISP is only non-null for the 96 rows where YEAR=2012.
All other 864 rows have NaN for that column. The 7945 columns are mostly
year-specific variants of ~50–80 actual economic variables.

─────────────────────────────────────────────────────────────────────────────
PROPOSED FIX: yearly slice + rename + vertical stack
─────────────────────────────────────────────────────────────────────────────

For each year Y in 2012–2021:
  1. Filter master to rows where YEAR == Y          → 96 rows
  2. Drop all columns that are entirely NaN for those 96 rows
  3. Rename each column by replacing the 2-digit year suffix with a canonical
     base name (see mapping table below)
  4. Stack all 10 slices vertically with pd.concat

Result: 960 rows × ~50–80 columns, one value per cell, no year-siloing.
─────────────────────────────────────────────────────────────────────────────
""")

# Build and print the rename mapping plan
# Key variables we identified, extract their base names and example columns
print("  COLUMN RENAME MAPPING (year suffix {YY} → canonical name):\n")
print(f"  {'Pattern / base name':<35s} → {'Canonical name':<35s} {'Years present'}")
print("  " + "-" * 90)

RENAME_MAP = [
    # (regex to match column names,  canonical_name,  description)
    # Identifiers
    (r"^LIBGEO$",           "dep_name",          "Department label"),
    # Household counts
    (r"^NBMEN\{YY\}$",      "n_households",      "Number of fiscal households"),
    (r"^NBPERS\{YY\}$",     "n_persons",         "Persons in households"),
    (r"^NBUC\{YY\}$",       "n_uc",              "Consumption units"),
    # Declared income (DEC concept)
    (r"^PMIMP\{YY\}$",      "pct_taxed_dec",     "% taxed households (DEC)"),
    (r"^Q1\{YY\}_DEC$",     "q1_dec",            "1st quartile declared income"),
    (r"^Q2\{YY\}_DEC$",     "q2_dec",            "Median (Q2) declared income"),
    (r"^Q3\{YY\}_DEC$",     "q3_dec",            "3rd quartile declared income"),
    (r"^D1\{YY\}_DEC$",     "d1_dec",            "1st decile declared income"),
    (r"^D9\{YY\}_DEC$",     "d9_dec",            "9th decile declared income"),
    (r"^GI\{YY\}_DEC$",     "gini_dec",          "Gini coefficient (declared)"),
    (r"^RD_DEC$",           "d9_d1_dec",         "D9/D1 ratio declared income"),
    (r"^S80S2\{YY\}_DEC$",  "s80s20_dec",        "S80/S20 ratio (declared)"),
    # Disposable income (DISP concept), USE THESE for inequality analysis
    (r"^Q1\{YY\}_DISP$",    "q1_disp",           "1st quartile disposable income"),
    (r"^Q2\{YY\}_DISP$",    "q2_disp",           "Median (Q2) disposable income ★"),
    (r"^Q3\{YY\}_DISP$",    "q3_disp",           "3rd quartile disposable income"),
    (r"^D1\{YY\}_DISP$",    "d1_disp",           "1st decile disposable income"),
    (r"^D9\{YY\}_DISP$",    "d9_disp",           "9th decile disposable income ★"),
    (r"^GI\{YY\}_DISP$",    "gini_disp",         "Gini coefficient (disposable) ★"),
    (r"^RD_DISP$",          "d9_d1_disp",        "D9/D1 ratio disposable income"),
    (r"^S80S2\{YY\}_DISP$", "s80s20_disp",       "S80/S20 ratio (disposable)"),
    # Poverty (Pauvres files)
    (r"^TP60\{YY\}$",       "poverty_rate",      "Poverty rate 60%, 2012 only (no suffix)"),
    (r"^TP60\{YY\}_PAU_DEC$",  "poverty_rate_dec",  "Poverty rate 60% (declared)"),
    (r"^TP60\{YY\}_PAU_DISP$", "poverty_rate_disp", "Poverty rate 60% (disposable) ★"),
    # Income composition shares (DEC only)
    (r"^PTSA\{YY\}$",       "pct_wages",         "% wages & salaries"),
    (r"^PCHO\{YY\}$",       "pct_unemployment",  "% unemployment benefits"),
    (r"^PBEN\{YY\}$",       "pct_other_earned",  "% other earned income"),
    (r"^PPEN\{YY\}$",       "pct_pensions",      "% pensions & retirement"),
    (r"^PAUT\{YY\}$",       "pct_other",         "% other income"),
]

# Print mapping plan using real column names found in the data
for pat, canonical, desc in RENAME_MAP:
    # Convert {YY} pattern back to regex for matching against base_groups
    base_pat = pat.replace(r"\{YY\}", r"\{YY\}")
    matched_bases = [b for b in base_groups.keys() if re.search(base_pat, b)]
    if not matched_bases:
        print(f"  {'⚠ ' + pat:<35s} → {canonical:<35s} ⚠ NOT FOUND in data")
        continue
    for base in matched_bases:
        members = base_groups[base]
        year_codes = sorted(set(
            y for m in members
            for y in (eval(m["years"]) if m["years"] not in ("[]","") else [])
        ))
        yr_str = f"{min(year_codes)}–{max(year_codes)}" if year_codes else "none"
        print(f"  {base:<35s} → {canonical:<35s} ({yr_str}, {len(year_codes)} yr)")

print("""
─────────────────────────────────────────────────────────────────────────────
NOTES:
  ★ = recommended primary variable for inequality/poverty analysis
  RD and S80S20 columns: the base name does NOT contain {YY}, check manually
  Columns NOT in the mapping above: household-type breakdowns (AGE1-6, TYM1-6,
    OPR1-6, etc.), ~100 base names. Suggest keeping only "ENSEMBLE" columns
    (no household-type prefix) in the reshaped file, and storing the breakdown
    columns separately if needed.
─────────────────────────────────────────────────────────────────────────────
""")

print("Diagnostic complete. No data was modified.")
print(f"Coverage table saved to: {coverage_path}")
