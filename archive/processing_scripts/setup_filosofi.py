"""
Régions Inégales — Filosofi Data Setup
Tasks: extract ZIPs, inspect structure, assess CSV conversion, report join keys.
"""

import os
import re
import zipfile
import pathlib
import traceback

import pandas as pd
import openpyxl

BASE_DIR = pathlib.Path("/home/crusie/3. Code/Régions Inégales/Base niveau administratif")
PROJECT_ROOT = pathlib.Path("/home/crusie/3. Code/Régions Inégales")

# ---------------------------------------------------------------------------
# Helper: detect year from filename
# ---------------------------------------------------------------------------
def extract_year(filename):
    m = re.search(r'(20\d{2})', filename)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Task 1 — Extract all ZIPs into annee_YYYY subfolders
# ---------------------------------------------------------------------------
print("=" * 70)
print("TASK 1 — Extracting ZIPs into year subfolders")
print("=" * 70)

zip_files = sorted(BASE_DIR.glob("*.zip"))
extracted_years = {}

for zf in zip_files:
    year = extract_year(zf.name)
    if not year:
        print(f"  WARNING: Could not detect year for {zf.name} — skipping")
        continue

    target_dir = BASE_DIR / f"annee_{year}"
    target_dir.mkdir(exist_ok=True)

    try:
        with zipfile.ZipFile(zf, 'r') as z:
            z.extractall(target_dir)
        extracted_years[year] = target_dir
        print(f"  OK  {zf.name}  →  annee_{year}/  ({len(z.namelist())} entries)")
    except Exception as e:
        print(f"  ERROR extracting {zf.name}: {e}")

print()


# ---------------------------------------------------------------------------
# Task 2 & 3 — Inspect + assess conversion feasibility
# ---------------------------------------------------------------------------
print("=" * 70)
print("TASKS 2 & 3 — Structural inspection + CSV conversion assessment")
print("=" * 70)

# Collect join-key info for final summary
join_key_summary = {}
conversion_report = {}  # year -> list of dicts

SAMPLE_ROWS = 3
POSSIBLE_JOIN_KEYS = {"dep", "codgeo", "code", "code_dep", "code_departement", "reg", "coddep"}


def detect_join_key(columns):
    for col in columns:
        if col.strip().lower() in POSSIBLE_JOIN_KEYS:
            return col.strip()
    return None


def inspect_csv(filepath, year):
    result = {"file": filepath.name, "type": "CSV", "sheets": None, "safe_to_convert": "already CSV"}
    try:
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                df = pd.read_csv(filepath, encoding=enc, sep=None, engine="python", nrows=SAMPLE_ROWS + 1)
                result["encoding"] = enc
                result["columns"] = list(df.columns)
                result["nrows_sample"] = len(df)
                result["sample"] = df.head(SAMPLE_ROWS).to_string(index=False)
                jk = detect_join_key(df.columns)
                result["join_key"] = jk
                break
            except UnicodeDecodeError:
                continue

        # Get full row count without reading everything
        try:
            full_df = pd.read_csv(filepath, encoding=result.get("encoding", "latin-1"), sep=None, engine="python")
            result["total_rows"] = len(full_df)
        except Exception:
            result["total_rows"] = "?"

    except Exception as e:
        result["error"] = str(e)
    return result


def inspect_excel(filepath, year):
    result = {"file": filepath.name, "type": "XLSX", "safe_to_convert": None}
    try:
        wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
        sheets = wb.sheetnames
        result["sheets"] = sheets
        result["sheet_count"] = len(sheets)
        result["safe_to_convert"] = "safe — 1 sheet" if len(sheets) == 1 else f"multi-sheet ({len(sheets)} sheets) — needs manual decision"
        wb.close()

        # Read with pandas for column + row info (first sheet)
        df_full = pd.read_excel(filepath, sheet_name=0, dtype=str)
        result["columns"] = list(df_full.columns)
        result["total_rows"] = len(df_full)
        result["sample"] = df_full.head(SAMPLE_ROWS).to_string(index=False)
        jk = detect_join_key(df_full.columns)
        result["join_key"] = jk

    except Exception as e:
        result["error"] = str(e)
        result["safe_to_convert"] = "ERROR — could not open"
    return result


for year in sorted(extracted_years.keys()):
    folder = extracted_years[year]
    print(f"\n{'─' * 60}")
    print(f"  YEAR {year}  |  {folder}")
    print(f"{'─' * 60}")

    all_files = [f for f in folder.rglob("*") if f.is_file()]
    csv_files = [f for f in all_files if f.suffix.lower() == ".csv"]
    xlsx_files = [f for f in all_files if f.suffix.lower() in (".xlsx", ".xls")]
    other_files = [f for f in all_files if f not in csv_files and f not in xlsx_files]

    print(f"  Total files: {len(all_files)}  (CSV: {len(csv_files)}, Excel: {len(xlsx_files)}, Other: {len(other_files)})")
    if other_files:
        print(f"  Other files: {[f.name for f in other_files]}")

    year_results = []
    year_join_keys = set()

    for f in sorted(csv_files + xlsx_files):
        print(f"\n  FILE: {f.relative_to(folder)}")
        if f.suffix.lower() == ".csv":
            r = inspect_csv(f, year)
        else:
            r = inspect_excel(f, year)

        year_results.append(r)

        if "error" in r:
            print(f"    ERROR: {r['error']}")
            continue

        print(f"    Type      : {r['type']}")
        if r.get("encoding"):
            print(f"    Encoding  : {r['encoding']}")
        if r.get("sheets") is not None:
            print(f"    Sheets    : {r['sheets']}")
        print(f"    Columns   : {r.get('columns', 'N/A')}")
        print(f"    Total rows: {r.get('total_rows', 'N/A')}")
        print(f"    Join key  : {r.get('join_key', '⚠ not found — check manually')}")
        print(f"    Conversion: {r.get('safe_to_convert', 'N/A')}")
        print(f"    Sample ({SAMPLE_ROWS} rows):")
        sample_text = r.get("sample", "N/A")
        for line in sample_text.splitlines():
            print(f"      {line}")

        if r.get("join_key"):
            year_join_keys.add(r["join_key"])

    conversion_report[year] = year_results
    join_key_summary[year] = year_join_keys if year_join_keys else {"⚠ NOT FOUND"}


# ---------------------------------------------------------------------------
# Final summary: join keys + conversion
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY — Join key consistency across years")
print("=" * 70)
for year in sorted(join_key_summary.keys()):
    print(f"  {year}:  {join_key_summary[year]}")

print("\n" + "=" * 70)
print("SUMMARY — CSV conversion assessment")
print("=" * 70)
for year in sorted(conversion_report.keys()):
    print(f"\n  {year}:")
    for r in conversion_report[year]:
        status = r.get("safe_to_convert", "N/A")
        err = f"  ← ERROR: {r['error']}" if "error" in r else ""
        print(f"    {r['file']:55s}  {status}{err}")

print()
print("Setup and inspection complete. No data was merged or modified.")
