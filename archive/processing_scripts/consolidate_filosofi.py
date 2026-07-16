"""
Régions Inégales - Filosofi Consolidation
Tasks 1–4: column inventory, wide CSV per year, stack all years, sanity check.
"""

import pathlib, sys
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT  = pathlib.Path("/home/crusie/3. Code/Régions Inégales")
BASE     = PROJECT / "Base niveau administratif"
OUT_DIR  = PROJECT / "filosofi_clean"
OUT_DIR.mkdir(exist_ok=True)

YEARS_XLS = [2012, 2013, 2014, 2015, 2016, 2017]
YEARS_CSV = [2018, 2019, 2020, 2021]
ALL_YEARS = YEARS_XLS + YEARS_CSV

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_dep_files_xls(year: int) -> list[pathlib.Path]:
    """Return sorted list of all *_DEP.xls/.xlsx files for a given year."""
    folder = BASE / f"annee_{year}"
    files = sorted([
        f for f in folder.rglob("*")
        if f.is_file()
        and f.name.upper().endswith(("_DEP.XLS", "_DEP.XLSX"))
        and f.suffix.lower() in (".xls", ".xlsx")
    ])
    return files


def find_dep_files_csv(year: int) -> list[pathlib.Path]:
    """Return sorted list of all *_DEP.csv files (non-meta) for a given year."""
    folder = BASE / f"annee_{year}"
    files = sorted([
        f for f in folder.glob("*.csv")
        if f.name.upper().endswith("_DEP.CSV")
        and not f.name.lower().startswith("meta_")
    ])
    return files


def _read_xls_sheet(filepath: pathlib.Path, sheet_name, engine: str) -> pd.DataFrame:
    """Read a single named sheet, auto-detecting the CODGEO header row."""
    raw = pd.read_excel(filepath, sheet_name=sheet_name, header=None,
                        engine=engine, dtype=str, nrows=15)
    codgeo_row = None
    for idx, row in raw.iterrows():
        if any(str(v).strip().upper() == "CODGEO" for v in row):
            codgeo_row = idx
            break
    if codgeo_row is None:
        raise ValueError(f"CODGEO row not found in sheet '{sheet_name}' of {filepath.name}")

    df = pd.read_excel(filepath, sheet_name=sheet_name, header=codgeo_row,
                       engine=engine, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")
    df = df[df["CODGEO"].notna()].copy()
    return df


def read_xls_ensemble(filepath: pathlib.Path) -> pd.DataFrame:
    """
    Read department-level data from an XLS/XLSX file.
    - Standard files (DEC, DISP, Pauvres): read the ENSEMBLE sheet.
    - TRDECILES files: no ENSEMBLE sheet, instead read all TRDEC_1…TRDEC_10
      sheets and merge them horizontally on CODGEO (mirrors the flat CSV layout
      used in 2018–2021).
    """
    ext = filepath.suffix.lower()
    engine = "xlrd" if ext == ".xls" else "openpyxl"

    # Discover sheet names without loading data
    if engine == "xlrd":
        import xlrd as _xlrd
        wb = _xlrd.open_workbook(str(filepath), on_demand=True)
        sheet_names = wb.sheet_names()
        wb.release_resources()
    else:
        import openpyxl as _openpyxl
        wb = _openpyxl.load_workbook(filepath, read_only=True)
        sheet_names = wb.sheetnames
        wb.close()

    if "ENSEMBLE" in sheet_names:
        return _read_xls_sheet(filepath, "ENSEMBLE", engine)

    # TRDECILES file: merge TRDEC_1 … TRDEC_10 on CODGEO
    trdec_sheets = [s for s in sheet_names if s.upper().startswith("TRDEC_")]
    if not trdec_sheets:
        raise ValueError(f"No ENSEMBLE or TRDEC_* sheets found in {filepath.name} "
                         f"(sheets: {sheet_names})")

    frames = []
    for sheet in sorted(trdec_sheets, key=lambda s: int(s.split("_")[-1])):
        frames.append(_read_xls_sheet(filepath, sheet, engine))

    # Merge all TRDEC sheets on CODGEO, each sheet adds different columns
    base = frames[0]
    for fr in frames[1:]:
        new_cols = ["CODGEO"] + [c for c in fr.columns if c not in base.columns]
        base = base.merge(fr[new_cols], on="CODGEO", how="outer")

    return base


def read_csv_dep(filepath: pathlib.Path) -> pd.DataFrame:
    """Read a semicolon-separated INSEE CSV with UTF-8 or Latin-1 fallback."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(filepath, sep=";", encoding=enc, dtype=str)
            if "CODGEO" in df.columns:
                return df
        except (UnicodeDecodeError, Exception):
            continue
    raise ValueError(f"Could not decode {filepath.name}")


def is_dept_codgeo(s: pd.Series) -> pd.Series:
    """True for 2-character department codes: 01–19, 21–95, 2A, 2B."""
    cleaned = s.str.strip().str.upper()
    return (cleaned.str.len() == 2) & (
        cleaned.str.match(r"^(0[1-9]|[1-9][0-9]|2[AB])$")
    )


def infer_concept(filename: str) -> str:
    """Extract a short income-concept tag from the filename (used as column suffix)."""
    n = filename.upper()
    if "TRDECILES" in n and "DISP" in n:   return "TR_DISP"
    if "TRDECILES" in n and "DEC"  in n:   return "TR_DEC"
    if "DISP" in n and ("PAUVRES" in n):    return "PAU_DISP"
    if  "DEC" in n and ("PAUVRES" in n):    return "PAU_DEC"
    if "DISP" in n:                         return "DISP"
    if  "DEC" in n:                         return "DEC"
    return "UNK"


def horizontal_merge(dfs: list[pd.DataFrame], source_names: list[str]) -> pd.DataFrame:
    """
    Merge a list of dataframes horizontally on CODGEO (outer join).

    Column-conflict resolution:
    - Same name, identical values  → keep one copy (truly redundant)
    - Same name, different values  → keep BOTH; rename the earlier copy with
      _{concept_A} suffix and the later copy with _{concept_B} suffix so that
      no data is silently lost (e.g. GI12_DEC vs GI12_DISP).

    LIBGEO (department label) is kept from the first file only.
    """
    SKIP_RENAME = {"CODGEO", "LIBGEO"}

    concepts = [infer_concept(n) for n in source_names]
    # track which concept contributed each column in the running base
    col_concept: dict[str, str] = {c: concepts[0] for c in dfs[0].columns
                                   if c not in SKIP_RENAME}

    base = dfs[0].copy()

    for df, src_name, concept in zip(dfs[1:], source_names[1:], concepts[1:]):
        non_id_cols  = [c for c in df.columns if c not in SKIP_RENAME]
        overlap      = [c for c in non_id_cols if c in base.columns]
        skip_libgeo  = ["LIBGEO"] if "LIBGEO" in df.columns else []

        rename_in_base: dict[str, str] = {}  # renames to apply to base before merge
        rename_in_new:  dict[str, str] = {}  # renames to apply to incoming df
        drop_from_new:  list[str]      = []  # cols to drop from incoming (redundant)

        for col in overlap:
            # fast equality check via aligned series
            check = base[["CODGEO", col]].merge(
                df[["CODGEO", col]], on="CODGEO", suffixes=("_a","_b"), how="outer"
            )
            if check[f"{col}_a"].equals(check[f"{col}_b"]):
                drop_from_new.append(col)   # identical, safe to drop
            else:
                orig_c = col_concept.get(col, "UNK")
                rename_in_base[col] = f"{col}_{orig_c}"
                rename_in_new[col]  = f"{col}_{concept}"

        # Apply renames to base
        if rename_in_base:
            base = base.rename(columns=rename_in_base)
            for old, new in rename_in_base.items():
                col_concept[new] = col_concept.pop(old, "UNK")
            print(f"    ℹ  {src_name}: {len(rename_in_base)} columns renamed with concept suffix "
                  f"(e.g. '{list(rename_in_base.values())[0]}')")

        # Build the slice of new df to merge
        incoming_cols = ["CODGEO"] + [
            c for c in df.columns
            if c not in SKIP_RENAME and c not in drop_from_new
        ]
        incoming = df[incoming_cols + skip_libgeo].copy()
        if rename_in_new:
            incoming = incoming.rename(columns=rename_in_new)
        for col in incoming.columns:
            if col not in SKIP_RENAME:
                col_concept[col] = concept

        # Keep LIBGEO only from the first file
        merge_cols = [c for c in incoming.columns if c != "LIBGEO"]
        base = base.merge(incoming[merge_cols], on="CODGEO", how="outer")

        if drop_from_new:
            print(f"    ✓  {src_name}: {len(drop_from_new)} truly-identical columns dropped")

    return base


# ---------------------------------------------------------------------------
# TASK 1, Column inventory
# ---------------------------------------------------------------------------
print("=" * 72)
print("TASK 1, COLUMN INVENTORY BY YEAR")
print("=" * 72)

# Store column sets per year for the presence/absence table later
year_columns: dict[int, list[str]] = {}

for year in ALL_YEARS:
    print(f"\n{'─'*60}")
    print(f"  YEAR {year}")
    print(f"{'─'*60}")

    try:
        if year in YEARS_XLS:
            files = find_dep_files_xls(year)
        else:
            files = find_dep_files_csv(year)

        all_cols_this_year: list[str] = []

        for f in files:
            try:
                if year in YEARS_XLS:
                    df = read_xls_ensemble(f)
                else:
                    df = read_csv_dep(f)

                dept_mask = is_dept_codgeo(df["CODGEO"])
                df_dept = df[dept_mask]

                print(f"\n  FILE : {f.name}")
                print(f"  Shape (all rows): {df.shape}  |  Dept rows: {len(df_dept)}")
                print(f"  Columns ({len(df.columns)}): {list(df.columns)}")
                print(f"  First 2 rows (dept-filtered):")
                print(df_dept.head(2).to_string(index=False))

                all_cols_this_year.extend([c for c in df.columns if c not in all_cols_this_year])

            except Exception as e:
                print(f"  ERROR reading {f.name}: {e}")

        year_columns[year] = all_cols_this_year

    except Exception as e:
        print(f"  FATAL for year {year}: {e}")


# ---------------------------------------------------------------------------
# TASK 2, Wide CSV per year (department level)
# ---------------------------------------------------------------------------
print("\n\n" + "=" * 72)
print("TASK 2, BUILDING filosofi_YYYY_dept.csv FOR EACH YEAR")
print("=" * 72)

year_dfs: dict[int, pd.DataFrame] = {}

for year in ALL_YEARS:
    print(f"\n  YEAR {year}")
    try:
        if year in YEARS_XLS:
            files = find_dep_files_xls(year)
            dfs   = [read_xls_ensemble(f) for f in files]
        else:
            files = find_dep_files_csv(year)
            dfs   = [read_csv_dep(f) for f in files]

        names = [f.name for f in files]
        merged = horizontal_merge(dfs, names)

        # Filter to department rows
        dept_mask = is_dept_codgeo(merged["CODGEO"])
        merged = merged[dept_mask].copy()
        merged["CODGEO"] = merged["CODGEO"].str.strip().str.upper()

        # Add YEAR column
        merged["YEAR"] = year

        # Sort columns: YEAR, CODGEO, rest alphabetically
        other_cols = sorted(c for c in merged.columns if c not in ("YEAR", "CODGEO"))
        merged = merged[["YEAR", "CODGEO"] + other_cols].reset_index(drop=True)

        out_path = OUT_DIR / f"filosofi_{year}_dept.csv"
        merged.to_csv(out_path, sep=";", encoding="utf-8", index=False)

        print(f"    Saved: {out_path.name}  ({len(merged)} rows × {len(merged.columns)} cols)")
        year_dfs[year] = merged

    except Exception as e:
        import traceback
        print(f"    ERROR for year {year}: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# TASK 3, Stack all years
# ---------------------------------------------------------------------------
print("\n\n" + "=" * 72)
print("TASK 3, STACKING ALL YEARS → filosofi_all_years.csv")
print("=" * 72)

all_frames = [year_dfs[y] for y in sorted(year_dfs.keys())]
master = pd.concat(all_frames, ignore_index=True, sort=False)

master_path = OUT_DIR / "filosofi_all_years.csv"
master.to_csv(master_path, sep=";", encoding="utf-8", index=False)

print(f"\n  Total rows    : {len(master)}")
print(f"  Total columns : {len(master.columns)}")
print(f"  Saved         : {master_path}")

# Column presence/absence table
print("\n  Column presence by year (showing non-CODGEO/YEAR/LIBGEO cols):")
all_cols_sorted = sorted(c for c in master.columns if c not in ("YEAR", "CODGEO", "LIBGEO"))
present_years   = sorted(year_dfs.keys())

# Print header
hdr = "  {:<30s}".format("COLUMN") + "".join(f" {y}" for y in present_years)
print(hdr)
print("  " + "-" * (30 + 5 * len(present_years)))

for col in all_cols_sorted:
    row_str = f"  {col:<30s}"
    for y in present_years:
        has_it = col in year_dfs[y].columns
        row_str += "    ✓" if has_it else "    ·"
    print(row_str)


# ---------------------------------------------------------------------------
# TASK 4, Sanity checks
# ---------------------------------------------------------------------------
print("\n\n" + "=" * 72)
print("TASK 4, SANITY CHECKS")
print("=" * 72)

# 4a, YEAR completeness
print("\n  Rows per year:")
year_counts = master["YEAR"].value_counts().sort_index()
for y, n in year_counts.items():
    gap = "  ← ⚠ MISSING?" if int(y) not in ALL_YEARS else ""
    print(f"    {y}: {n} rows{gap}")
missing_years = set(range(2012, 2022)) - set(int(y) for y in year_counts.index)
if missing_years:
    print(f"  ⚠ Missing years: {sorted(missing_years)}")
else:
    print("  ✓ All years 2012–2021 present")

# 4b, CODGEO values
print("\n  Unexpected CODGEO values (not standard 2-char dept code):")
bad = master[~is_dept_codgeo(master["CODGEO"])]["CODGEO"].unique()
if len(bad) == 0:
    print("  ✓ All CODGEO values are valid 2-character department codes")
else:
    print(f"  ⚠ Unexpected: {sorted(bad)[:30]}")

# 4c, Departments that appear fewer than 10 times
print("\n  Departments with < 10 year-appearances (expected max 10):")
dept_counts = master["CODGEO"].value_counts()
sparse = dept_counts[dept_counts < 10]
if len(sparse) == 0:
    print("  ✓ All departments appear in 10 years")
else:
    for code, count in sparse.sort_index().items():
        print(f"    {code}: {count} year(s)")

# 4d, Key variable columns search
# INSEE naming note:
#   Median income = Q2XX (second quartile), not "MED", search for "Q2"
#   Poverty rate  = TP60XX (overall rate in Pauvres files)
#   Gini          = GIXX
print("\n  Key variables search:")
print("  (Note: INSEE calls median income Q2, not MED, e.g. Q212 for 2012)")
for pattern in ("^Q2[0-9]", "TP60[0-9]", "^GI[0-9]"):
    import re as _re
    matches = [c for c in master.columns if _re.search(pattern, c, _re.IGNORECASE)]
    if not matches:
        print(f"  ⚠  No columns containing '{pattern}'")
        continue
    print(f"\n  Columns containing '{pattern}':")
    for col in sorted(matches):
        non_null = master[col].notna().sum()
        years_present = master.loc[master[col].notna(), "YEAR"].unique()
        print(f"    {col:<35s}  non-null: {non_null:>5}  "
              f"years: {sorted(int(y) for y in years_present)}")

print("\n\nConsolidation complete.")
print(f"Main output: {master_path}")
