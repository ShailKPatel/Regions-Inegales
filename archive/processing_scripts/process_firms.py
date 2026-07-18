#!/usr/bin/env python3
"""
Tasks 1–5: Extract and process INSEE SIDE firm creations dataset.
Reads data directly from ZIP (streaming) to avoid extracting the 4.6 GB CSV.
"""

import os
import zipfile
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ZIP_PATH  = os.path.join(PROJECT, "d'entreprises/DS_SIDE_CREA_DEP_REG_NAT_2024_CSV_FR.zip")
RAW_DIR   = os.path.join(PROJECT, "firms_raw")
CLEAN_DIR = os.path.join(PROJECT, "firms_clean")

os.makedirs(RAW_DIR,   exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

DATA_FILE = "DS_SIDE_CREA_DEP_REG_NAT_2024_data.csv"
META_FILE = "DS_SIDE_CREA_DEP_REG_NAT_2024_metadata.csv"

# ── Metropolitan department codes (96 depts, preserving leading zeros) ─────────
METRO_DEPS = (
    {f"{i:02d}" for i in range(1, 20)}    # 01–19
    | {f"{i:02d}" for i in range(21, 96)} # 21–95 (20 was split into 2A/2B)
    | {"2A", "2B"}
)
assert len(METRO_DEPS) == 96, f"Expected 96 deps, got {len(METRO_DEPS)}"

# ── Sector groupings (A21 codes) ───────────────────────────────────────────────
INDUSTRY     = {"B", "C", "D", "E"}          # mining, manufacturing, energy, water
CONSTRUCTION = {"F"}
TRADE        = {"G"}
SERVICES     = {"H", "I", "J", "K", "L", "M", "N", "P", "Q", "R", "S"}

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1, Extract and inspect
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("TASK 1, Extract and inspect")
print("=" * 65)

zf = zipfile.ZipFile(ZIP_PATH)

print("\nFiles in ZIP:")
for info in zf.infolist():
    print(f"  {info.filename}  ({info.file_size / 1e6:.1f} MB uncompressed)")

# Extract metadata to firms_raw/
print(f"\nExtracting metadata → {RAW_DIR}/")
zf.extract(META_FILE, RAW_DIR)

meta_path = os.path.join(RAW_DIR, META_FILE)
meta = pd.read_csv(meta_path, sep=";", encoding="utf-8")
print(f"Metadata shape: {meta.shape}")
print("\nMetadata (varmod) contents:")
print(meta.to_string(index=False))

# Peek at data file
print(f"\n\nData file: {DATA_FILE}")
print(f"  Uncompressed size: {zf.getinfo(DATA_FILE).file_size / 1e9:.2f} GB")
print("  First 5 rows:")
with zf.open(DATA_FILE) as fh:
    peek = pd.read_csv(fh, sep=";", encoding="utf-8", nrows=5, dtype=str)
print(f"  Columns: {list(peek.columns)}")
print(peek.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# TASK 2, Filter to department level
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("TASK 2, Geographic level inspection & department filter")
print("=" * 65)

print("\nReading data in 1 M-row chunks; keeping GEO_OBJECT='DEP' + SIDE_MEASURE='BURE'...")

chunks      = []
total_rows  = 0
geo_values  = set()
shown_geo_vals = False

with zf.open(DATA_FILE) as fh:
    reader = pd.read_csv(
        fh, sep=";", encoding="utf-8",
        chunksize=1_000_000,
        dtype={
            "GEO": str, "GEO_OBJECT": str, "ACTIVITY": str,
            "FREQ": str, "LEGAL_FORM": str, "SIDE_MEASURE": str,
            "TIME_PERIOD": str,
        },
    )
    for i, chunk in enumerate(reader):
        total_rows += len(chunk)

        # Collect all GEO_OBJECT levels seen (from first chunk is enough)
        geo_values.update(chunk["GEO_OBJECT"].unique())

        if not shown_geo_vals and len(geo_values) >= 2:
            print(f"\nGEO_OBJECT unique values seen so far: {sorted(geo_values)}")
            shown_geo_vals = True

        if i % 5 == 0:
            print(f"  Processed {total_rows:,} rows so far...")

        # Keep only: firm creations (BURE) at department level
        chunk = chunk[chunk["SIDE_MEASURE"] == "BURE"]
        chunk = chunk[chunk["GEO_OBJECT"] == "DEP"]
        # Metropolitan departments only
        chunk = chunk[chunk["GEO"].isin(METRO_DEPS)]

        if len(chunk) > 0:
            chunks.append(chunk)

print(f"\nTotal rows in raw file: {total_rows:,}")
print(f"All GEO_OBJECT values: {sorted(geo_values)}")

df = pd.concat(chunks, ignore_index=True)
df["OBS_VALUE"]   = pd.to_numeric(df["OBS_VALUE"], errors="coerce").fillna(0)
df["TIME_PERIOD"] = df["TIME_PERIOD"].astype(int)

print(f"\nRows after filtering (metro DEP + BURE): {len(df):,}")
print(f"Unique years:        {sorted(df['TIME_PERIOD'].unique())}")
print(f"Unique departments:  {len(df['GEO'].unique())} → {sorted(df['GEO'].unique())}")
print(f"Unique ACTIVITY codes: {sorted(df['ACTIVITY'].unique())}")
print(f"Unique LEGAL_FORM codes: {sorted(df['LEGAL_FORM'].unique())}")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 3, Aggregate to panel format
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("TASK 3, Aggregate to (dep × year) panel")
print("=" * 65)

# Trim to Filosofi window
df = df[df["TIME_PERIOD"].between(2012, 2021)]

def agg(sub_df):
    return (
        sub_df.groupby(["GEO", "TIME_PERIOD"])["OBS_VALUE"]
        .sum()
        .reset_index()
        .rename(columns={"GEO": "dep_code", "TIME_PERIOD": "year"})
    )

# Total: both ACTIVITY and LEGAL_FORM are _T (prevents double-counting)
total = agg(df[(df["ACTIVITY"] == "_T") & (df["LEGAL_FORM"] == "_T")])
total = total.rename(columns={"OBS_VALUE": "total_firm_creations"})

# Legal form breakdown, use ACTIVITY == "_T" (already the total across sectors)
act_t = df[df["ACTIVITY"] == "_T"]

lf_indiv = agg(act_t[act_t["LEGAL_FORM"] == "10"]   ).rename(columns={"OBS_VALUE": "creations_individual"})
lf_sarl  = agg(act_t[act_t["LEGAL_FORM"] == "54"]   ).rename(columns={"OBS_VALUE": "creations_sarl"})
lf_sas   = agg(act_t[act_t["LEGAL_FORM"] == "57"]   ).rename(columns={"OBS_VALUE": "creations_sas"})
lf_other = agg(act_t[act_t["LEGAL_FORM"] == "OTH_SIDE"]).rename(columns={"OBS_VALUE": "creations_other_legal"})

# Sector breakdown, use LEGAL_FORM == "_T" (already the total across legal forms)
lf_t = df[df["LEGAL_FORM"] == "_T"]

sec_industry     = agg(lf_t[lf_t["ACTIVITY"].isin(INDUSTRY)]    ).rename(columns={"OBS_VALUE": "creations_sector_industry"})
sec_construction = agg(lf_t[lf_t["ACTIVITY"] == "F"]            ).rename(columns={"OBS_VALUE": "creations_sector_construction"})
sec_trade        = agg(lf_t[lf_t["ACTIVITY"] == "G"]            ).rename(columns={"OBS_VALUE": "creations_sector_trade"})
sec_services     = agg(lf_t[lf_t["ACTIVITY"].isin(SERVICES)]    ).rename(columns={"OBS_VALUE": "creations_sector_services"})

# Merge all into panel
panel = total.copy()
for sub in [lf_indiv, lf_sarl, lf_sas, lf_other,
            sec_industry, sec_construction, sec_trade, sec_services]:
    panel = panel.merge(sub, on=["dep_code", "year"], how="left")

panel["dep_code"] = panel["dep_code"].astype(str)
panel["year"]     = panel["year"].astype(int)
panel = panel.sort_values(["dep_code", "year"]).reset_index(drop=True)

print(f"\nPanel shape: {panel.shape}  (expect 960 = 96 × 10)")
print(f"Columns: {list(panel.columns)}")
print(f"\nFirst 10 rows:")
print(panel.head(10).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# TASK 4, Sanity checks
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("TASK 4, Sanity checks")
print("=" * 65)

# National totals per year
print("\nTotal firm creations per year (sum across 96 metro depts):")
yearly = panel.groupby("year")["total_firm_creations"].sum()
for yr, val in yearly.items():
    flag = "  *** OUTSIDE EXPECTED RANGE ***" if val < 400_000 or val > 1_500_000 else ""
    print(f"  {yr}: {val:>10,.0f}{flag}")

# Top / bottom 5 in 2019
y2019 = panel[panel["year"] == 2019].sort_values("total_firm_creations", ascending=False)
print("\nTop 5 departments by total creations in 2019:")
print(y2019[["dep_code", "total_firm_creations"]].head(5).to_string(index=False))
print("\nBottom 5 departments by total creations in 2019:")
print(y2019[["dep_code", "total_firm_creations"]].tail(5).to_string(index=False))

# Zero creations
zeros = panel[panel["total_firm_creations"] == 0]
print(f"\nDepartment-years with zero total creations: {len(zeros)}")
if len(zeros) > 0:
    print(zeros[["dep_code", "year", "total_firm_creations"]].to_string(index=False))

# Null counts
print("\nNull count per column:")
print(panel.isnull().sum().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# TASK 5, Save and update DATA_SOURCES.md
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("TASK 5, Save firms_panel.csv & update DATA_SOURCES.md")
print("=" * 65)

out_path = os.path.join(CLEAN_DIR, "firms_panel.csv")
panel.to_csv(out_path, sep=";", encoding="utf-8", index=False)
print(f"\nSaved: {out_path}")
print(f"Shape: {panel.shape}")

data_sources_path = os.path.join(PROJECT, "DATA_SOURCES.md")
append_block = """
## 2. Firm Creations by Department (2012–2021)

- **Full name**: Créations d'entreprises et d'établissements, SIDE system
- **Producer**: INSEE (Institut national de la statistique et des études économiques)
- **Coverage**: Metropolitan France (96 departments), 2012–2021 (trimmed from 2012–2024)
- **Geographic level used**: Department
- **Variables**: Total firm creations, breakdown by legal form and broad sector
- **Sector classification**: A21 (aggregated to 4 broad groups for analysis)
- **License**: Free reuse, citation required
- **Citation**: INSEE, SIDE, "Démographie des entreprises : créations d'entreprises
  et d'établissements de 2012 à 2024", available at insee.fr
- **Downloaded from**: https://www.insee.fr/fr/statistiques/8557644
- **File used**: DS_SIDE_CREA_DEP_REG_NAT_2024_CSV_FR.zip
  (Données nationales, régionales et départementales, A21 breakdown)
- **Note on legal form**: LEGAL_FORM code 10 = "Entrepreneur individuel" covers ALL
  individual entrepreneurs (both micro/auto-entrepreneurs and other individuals).
  The SIDE dataset does not provide a sub-code to distinguish micro-entrepreneurs
  separately; `creations_individual` therefore represents the combined total.
"""

with open(data_sources_path, "a", encoding="utf-8") as f:
    f.write(append_block)
print(f"\nAppended firm creations block to DATA_SOURCES.md")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("NOTE ON SCHEMA ADJUSTMENT")
print("=" * 65)
print("""
The requested column 'creations_micro' (micro-entrepreneurs only) was replaced
by 'creations_individual' (all individual entrepreneurs, LEGAL_FORM code 10).

Reason: INSEE's SIDE dataset groups all individual entrepreneurs under a single
code (10 = 'Entrepreneur individuel'); there is no sub-code for micro-entrepreneurs
vs. other individuals in this file. The distinction would require INSEE's SIRENE
microdata or a separate publication.
""")

zf.close()
print("Done.\n")
