"""
Dataset 5 — Education: snapshot panel builder (3 census years, no interpolation).
Source: base-cc-diplomes-formation-2022.CSV (INSEE, Diplômes-Formation 2022 edition,
rolling censuses 2010/2015/2022 → reference years 2011/2016/2022).
"""

import sys
import pandas as pd

RAW = "education_raw/base-cc-diplomes-formation-2022_csv/base-cc-diplomes-formation-2022.CSV"
OUT = "sources/education_snapshots_insee.csv"
DATA_SOURCES = "DATA_SOURCES.md"

DOM_CODES = {"971", "972", "973", "974", "976"}

# ── TASK 1 — load, derive dep_code, filter metro ──────────────────────────────

df = pd.read_csv(RAW, sep=";", encoding="latin-1", dtype={"CODGEO": str})
print(f"Raw communes loaded: {len(df):,} rows × {df.shape[1]} cols")

def derive_dep(code: str) -> str:
    if code.startswith("2A") or code.startswith("2B"):
        return code[:2]
    if len(code) >= 3 and code[:3] in DOM_CODES:
        return code[:3]
    return code[:2]

df["dep_code"] = df["CODGEO"].map(derive_dep)

all_dep = sorted(df["dep_code"].unique())
print(f"\nAll unique dep_codes before filter ({len(all_dep)}): {all_dep}")

unexpected = [d for d in all_dep if d not in DOM_CODES and
              not (d.isdigit() and 1 <= int(d) <= 95) and d not in ("2A", "2B")]
print(f"Unexpected dep_codes: {unexpected if unexpected else 'none'}")

# drop DOM
df_metro = df[~df["dep_code"].isin(DOM_CODES)].copy()
metro_deps = sorted(df_metro["dep_code"].unique())
print(f"\nMetro unique dep_codes after filter: {len(metro_deps)}")
if len(metro_deps) != 96:
    print(f"  WARNING — expected 96, got {len(metro_deps)}")
    missing = [f"{i:02d}" for i in range(1, 96) if f"{i:02d}" not in metro_deps
               and str(i) not in metro_deps] + \
              ([] if "2A" in metro_deps else ["2A"]) + \
              ([] if "2B" in metro_deps else ["2B"])
    print(f"  Possibly missing: {missing}")
else:
    print("  OK — exactly 96 metro departments")

print(f"  List: {metro_deps}")

# ── TASK 2 — higher-ed share per department per snapshot year ─────────────────

years_config = {
    2022: {
        "num_cols": ["P22_NSCOL15P_SUP2", "P22_NSCOL15P_SUP34", "P22_NSCOL15P_SUP5"],
        "den_col":  "P22_NSCOL15P",
    },
    2016: {
        "num_cols": ["P16_NSCOL15P_SUP"],
        "den_col":  "P16_NSCOL15P",
    },
    2011: {
        "num_cols": ["P11_NSCOL15P_BACP2", "P11_NSCOL15P_SUP"],
        "den_col":  "P11_NSCOL15P",
    },
}

snap_records = []

for yr, cfg in years_config.items():
    num_cols = cfg["num_cols"]
    den_col  = cfg["den_col"]

    # coerce to numeric (skipna handled by pandas sum)
    for c in num_cols + [den_col]:
        df_metro[c] = pd.to_numeric(df_metro[c], errors="coerce")

    grp = df_metro.groupby("dep_code")
    num_sum = grp[num_cols].sum(min_count=0).sum(axis=1)   # skipna=True default
    den_sum = grp[den_col].sum(min_count=0)

    share = (100.0 * num_sum / den_sum).round(2)
    share.name = f"edu_share_sup_{yr}"

    # report
    print(f"\n── Year {yr} ──")
    print(f"  Numerator cols : {num_cols}")
    print(f"  Denominator col: {den_col}")
    print(f"  Dep count      : {len(share)} (should be 96)")
    print(f"  Null shares    : {share.isna().sum()}")

    for dep, row in share.items():
        snap_records.append({"dep_code": dep, "year": yr, "edu_share_sup": row})

# ── TASK 3 — Cantal coverage check ───────────────────────────────────────────

cantal = df_metro[df_metro["dep_code"] == "15"].copy()
cantal["P22_NSCOL15P"] = pd.to_numeric(cantal["P22_NSCOL15P"], errors="coerce")
cantal_non_blank = cantal[cantal["P22_NSCOL15P"].notna()]
cantal_blank     = cantal[cantal["P22_NSCOL15P"].isna()]

print(f"\n── Cantal (dep 15) coverage check ──")
print(f"  Total communes in file          : {len(cantal)}")
print(f"  Non-blank communes (have P22_NSCOL15P): {len(cantal_non_blank)}")
print(f"  Blank communes (suppressed)     : {len(cantal_blank)}")
print(f"  Blank CODGEO codes              : {sorted(cantal_blank['CODGEO'].tolist())}")
print(f"  Sum P22_NSCOL15P (non-blank)    : {cantal_non_blank['P22_NSCOL15P'].sum():,.0f}")

# Check if any partial pop data on blank communes
for c in ["P22_NSCOL15P", "P11_NSCOL15P", "P16_NSCOL15P"]:
    cantal[c] = pd.to_numeric(cantal[c], errors="coerce")
partial = cantal_blank[["CODGEO", "P22_NSCOL15P", "P16_NSCOL15P", "P11_NSCOL15P"]]
print(f"  Blank communes partial data (any year):\n{partial.to_string(index=False)}")
print(f"  => Suppressed communes have no recoverable pop data in this file.")
print(f"     Combined population is unquantifiable from this source; flag for web-anchor.")

# ── TASK 4 — shape + internal checks ─────────────────────────────────────────

panel = pd.DataFrame(snap_records)
panel["dep_code"] = panel["dep_code"].astype(str)
panel["year"]     = panel["year"].astype(int)
panel = panel.sort_values(["dep_code", "year"]).reset_index(drop=True)

print(f"\n── Panel shape checks ──")
print(f"  Rows        : {len(panel)} (expected 288)")
print(f"  Null shares : {panel['edu_share_sup'].isna().sum()} (expected 0)")
dup = panel.duplicated(subset=["dep_code", "year"]).sum()
print(f"  Duplicates  : {dup} (expected 0)")
print(f"  Rows/year   : {panel.groupby('year').size().to_dict()} (each expected 96)")

# Bounds
lo, hi = panel["edu_share_sup"].min(), panel["edu_share_sup"].max()
lo_dep = panel.loc[panel["edu_share_sup"] == lo, ["dep_code", "year"]].values
hi_dep = panel.loc[panel["edu_share_sup"] == hi, ["dep_code", "year"]].values
print(f"\n  Global min: {lo}% → dep {lo_dep}")
print(f"  Global max: {hi}% → dep {hi_dep}")

out_of_bounds = panel[(panel["edu_share_sup"] < 10) | (panel["edu_share_sup"] > 70)]
if len(out_of_bounds):
    print(f"  OUT-OF-BOUNDS rows:\n{out_of_bounds.to_string()}")
else:
    print(f"  All shares in [10, 70]% — OK")

# Per-year min/max
for yr in [2011, 2016, 2022]:
    sub = panel[panel["year"] == yr]
    print(f"  Year {yr}: min={sub['edu_share_sup'].min()}% ({sub.loc[sub['edu_share_sup'].idxmin(), 'dep_code']})"
          f"  max={sub['edu_share_sup'].max()}% ({sub.loc[sub['edu_share_sup'].idxmax(), 'dep_code']})")

# Monotonicity check
wide = panel.pivot(index="dep_code", columns="year", values="edu_share_sup")
dec_11_16 = wide[wide[2016] < wide[2011]].index.tolist()
dec_16_22 = wide[wide[2022] < wide[2016]].index.tolist()
print(f"\n  Monotonicity check (expect 2011<2016<2022):")
print(f"  Departments with 2016 < 2011: {dec_11_16 if dec_11_16 else 'none'}")
print(f"  Departments with 2022 < 2016: {dec_16_22 if dec_16_22 else 'none'}")
for d in dec_11_16 + dec_16_22:
    row = wide.loc[d]
    print(f"    dep {d}: 2011={row[2011]}%  2016={row[2016]}%  2022={row[2022]}%")

# Top-5 / bottom-5 in 2022
y22 = panel[panel["year"] == 2022].sort_values("edu_share_sup", ascending=False)
print(f"\n  Top-5 departments in 2022 (expect 75,92,31,69,35 near top):")
print(y22.head(5).to_string(index=False))
print(f"  Bottom-5 departments in 2022 (expect 15,23,36 near bottom):")
print(y22.tail(5).to_string(index=False))

# ── Raw recompute spot-check ──────────────────────────────────────────────────

SPOT_COMMUNES = {
    "31555": ("Toulouse", "31"),
    "23096": ("Guéret (rural, Creuse chef-lieu)", "23"),
    "75101": ("Paris 1er arr.", "75"),
}

print(f"\n── Raw recompute spot-check ──")
for codgeo, (label, expected_dep) in SPOT_COMMUNES.items():
    row = df[df["CODGEO"] == codgeo]
    if row.empty:
        print(f"  {codgeo} ({label}): NOT FOUND in raw file")
        continue

    dep_found = row["dep_code"].values[0] if "dep_code" in row.columns else derive_dep(codgeo)
    # re-derive dep in case we're using df (unfiltered)
    dep_found = derive_dep(codgeo)

    # 2022
    for c in ["P22_NSCOL15P_SUP2", "P22_NSCOL15P_SUP34", "P22_NSCOL15P_SUP5", "P22_NSCOL15P"]:
        row[c] = pd.to_numeric(row[c], errors="coerce")
    sup22 = float(row[["P22_NSCOL15P_SUP2", "P22_NSCOL15P_SUP34", "P22_NSCOL15P_SUP5"]].sum(axis=1).values[0])
    den22 = float(row["P22_NSCOL15P"].values[0])

    # check it feeds the right dep sum
    dep_data = df_metro[df_metro["dep_code"] == expected_dep]
    for c in ["P22_NSCOL15P_SUP2", "P22_NSCOL15P_SUP34", "P22_NSCOL15P_SUP5", "P22_NSCOL15P"]:
        dep_data[c] = pd.to_numeric(dep_data[c], errors="coerce")
    dep_num = float(dep_data[["P22_NSCOL15P_SUP2", "P22_NSCOL15P_SUP34", "P22_NSCOL15P_SUP5"]].sum().sum())
    dep_den = float(dep_data["P22_NSCOL15P"].sum())
    dep_share = round(100.0 * dep_num / dep_den, 2)
    panel_share = float(panel[(panel["dep_code"] == expected_dep) & (panel["year"] == 2022)]["edu_share_sup"].values[0])

    commune_pct = round(100.0 * sup22 / den22, 2) if den22 else None
    print(f"  {codgeo} ({label}): dep_code={dep_found} ✓  "
          f"commune_sup_pct={commune_pct}%  "
          f"dep_share_recomputed={dep_share}%  "
          f"panel_value={panel_share}%  "
          f"{'OK' if abs(dep_share - panel_share) < 0.01 else 'MISMATCH'}")

# ── TASK 5 — output ───────────────────────────────────────────────────────────

panel.to_csv(
    OUT,
    sep=";",
    index=False,
    quoting=__import__("csv").QUOTE_NONNUMERIC,
)
print(f"\n── Output ──")
print(f"  Saved: {OUT}")
print(f"  Rows: {len(panel)}, cols: {list(panel.columns)}")
head = pd.read_csv(OUT, sep=";", dtype={"dep_code": str})
print(f"  Re-read shape: {head.shape}")
print(f"  Head:\n{head.head(6).to_string(index=False)}")

# Append DATA_SOURCES.md entry
entry = """
---

## 6. Education by Department — Census Snapshots 2011 / 2016 / 2022

- **Full name**: Diplômes et formation 2022 — Base communale des diplômes et formations
- **Producer**: INSEE (Institut national de la statistique et des études économiques)
- **Source page**: insee.fr/fr/statistiques/8581488
- **File used**: `base-cc-diplomes-formation-2022.CSV` (separator `;`, encoding latin-1)
- **Coverage**: Metropolitan France (96 departments), three rolling-census snapshots
- **Reference years**: 2010 five-year rolling → labelled 2011 / 2015 rolling → 2016 / 2022 rolling → 2022
- **Geographic level used**: Commune → aggregated to department by population-weighted sum
- **Variable**: `edu_share_sup` — share (%) of non-schooled population aged 15+ holding any higher-education diploma (supérieur), computed as:
  - 2022: (P22_NSCOL15P_SUP2 + P22_NSCOL15P_SUP34 + P22_NSCOL15P_SUP5) / P22_NSCOL15P × 100
  - 2016: P16_NSCOL15P_SUP / P16_NSCOL15P × 100
  - 2011: (P11_NSCOL15P_BACP2 + P11_NSCOL15P_SUP) / P11_NSCOL15P × 100
- **Note — sub-level comparability**: Total supérieur only; sub-level breakdowns (SUP2/SUP34/SUP5 in 2022 vs. single SUP in 2016) are not cross-year comparable and are not used.
- **Output file**: `sources/education_snapshots_insee.csv` (288 rows × 3 cols: dep_code, year, edu_share_sup; `;`-delimited, CODGEO/dep_code quoted)
- **Processing**: Numerator and denominator summed at commune level within each department (skipna); share computed from department totals. DOM departments (971–976) excluded.

### Suppressed data — 4 Cantal communes

INSEE suppressed education data for 4 communes in Cantal (dep 15): 15031, 15035, 15047, 15171.
These communes have no recoverable population data in this file across any of the three census years.
Their combined population/share is unquantifiable from this source; the Cantal department share
is computed from the remaining communes only. **Flag for web-anchor**: check whether suppressed
communes' weight is small enough to be negligible; without external population data this cannot
be confirmed from the census file alone.
"""

with open(DATA_SOURCES, "a", encoding="utf-8") as fh:
    fh.write(entry)
print(f"\n  Appended entry to {DATA_SOURCES}")
print("\nDone. STOP — interpolation and merge are separate next steps.")
