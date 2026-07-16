"""
Régions Inégales - Filosofi Data Setup v2
Fixes: nested ZIP extraction for 2012/2014/2015, xlrd for .xls files, 2013 error.
"""

import os
import re
import zipfile
import pathlib
import xlrd

import pandas as pd
import openpyxl

BASE_DIR = pathlib.Path("/home/crusie/3. Code/Régions Inégales/Base niveau administratif")
PROJECT_ROOT = pathlib.Path("/home/crusie/3. Code/Régions Inégales")

SAMPLE_ROWS = 3
POSSIBLE_JOIN_KEYS = {"dep", "codgeo", "code", "code_dep", "code_departement", "reg", "coddep", "CODGEO", "DEP"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_year(filename):
    m = re.search(r'(20\d{2})', str(filename))
    return m.group(1) if m else None

def detect_join_key(columns):
    for col in columns:
        if col.strip().lower() in {k.lower() for k in POSSIBLE_JOIN_KEYS}:
            return col.strip()
    return None

def inspect_csv(filepath):
    result = {"file": filepath.name, "type": "CSV", "safe_to_convert": "already CSV"}
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(filepath, encoding=enc, sep=None, engine="python")
            result["encoding"] = enc
            result["columns"] = list(df.columns)
            result["total_rows"] = len(df)
            result["sample"] = df.head(SAMPLE_ROWS).to_string(index=False)
            result["join_key"] = detect_join_key(df.columns)
            return result
        except UnicodeDecodeError:
            continue
        except Exception as e:
            result["error"] = str(e)
            return result
    result["error"] = "Could not decode with utf-8, latin-1, or cp1252"
    return result

def inspect_xls(filepath):
    """Handle old .xls format using xlrd."""
    result = {"file": filepath.name, "type": "XLS (old)", "safe_to_convert": None}
    try:
        wb = xlrd.open_workbook(str(filepath))
        sheets = wb.sheet_names()
        result["sheets"] = sheets
        result["sheet_count"] = len(sheets)
        result["safe_to_convert"] = "safe, 1 sheet" if len(sheets) == 1 else f"multi-sheet ({len(sheets)}), needs manual decision"

        df = pd.read_excel(filepath, sheet_name=0, engine="xlrd", dtype=str)
        result["columns"] = list(df.columns)
        result["total_rows"] = len(df)
        result["sample"] = df.head(SAMPLE_ROWS).to_string(index=False)
        result["join_key"] = detect_join_key(df.columns)
    except Exception as e:
        result["error"] = str(e)
        result["safe_to_convert"] = "ERROR, could not open"
    return result

def inspect_xlsx(filepath):
    """Handle new .xlsx format using openpyxl."""
    result = {"file": filepath.name, "type": "XLSX", "safe_to_convert": None}
    try:
        wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
        sheets = wb.sheetnames
        result["sheets"] = sheets
        result["sheet_count"] = len(sheets)
        result["safe_to_convert"] = "safe, 1 sheet" if len(sheets) == 1 else f"multi-sheet ({len(sheets)}), needs manual decision"
        wb.close()

        df = pd.read_excel(filepath, sheet_name=0, engine="openpyxl", dtype=str)
        result["columns"] = list(df.columns)
        result["total_rows"] = len(df)
        result["sample"] = df.head(SAMPLE_ROWS).to_string(index=False)
        result["join_key"] = detect_join_key(df.columns)
    except Exception as e:
        result["error"] = str(e)
        result["safe_to_convert"] = "ERROR, could not open"
    return result


# ---------------------------------------------------------------------------
# Task 1b, Extract nested ZIPs for years 2012, 2014, 2015
# ---------------------------------------------------------------------------
print("=" * 70)
print("TASK 1b, Extracting nested ZIPs (2012, 2014, 2015)")
print("=" * 70)

for year in ["2012", "2014", "2015"]:
    year_dir = BASE_DIR / f"annee_{year}"
    nested_zips = list(year_dir.rglob("*.zip"))
    if not nested_zips:
        print(f"  {year}: no nested ZIPs found (already extracted?)")
        continue
    print(f"\n  {year}: found {len(nested_zips)} nested ZIP(s)")
    for nz in sorted(nested_zips):
        extract_to = nz.parent / nz.stem
        extract_to.mkdir(exist_ok=True)
        try:
            with zipfile.ZipFile(nz, 'r') as z:
                z.extractall(extract_to)
            print(f"    OK  {nz.name}  →  {extract_to.name}/  ({len(z.namelist())} entries)")
        except Exception as e:
            print(f"    ERROR  {nz.name}: {e}")

print()

# ---------------------------------------------------------------------------
# Task 2 & 3, Inspect all years
# ---------------------------------------------------------------------------
print("=" * 70)
print("TASKS 2 & 3, Structural inspection + CSV conversion assessment")
print("=" * 70)

all_years = sorted([d.name.replace("annee_", "") for d in BASE_DIR.iterdir()
                    if d.is_dir() and d.name.startswith("annee_")])

join_key_summary = {}
conversion_report = {}

# Also flag 2013 immediately
print(f"\n{'─' * 60}")
print(f"  YEAR 2013 , ⚠ ZIP CORRUPTED")
print(f"{'─' * 60}")
print(f"  ERROR: indic-struct-distrib-revenu-2013-SUPRA.zip is not a valid ZIP file.")
print(f"  ACTION NEEDED: Please re-download the 2013 file from:")
print(f"    https://www.insee.fr/fr/statistiques/2388413")
join_key_summary["2013"] = {"⚠ MISSING, ZIP corrupted, must re-download"}
conversion_report["2013"] = [{"file": "indic-struct-distrib-revenu-2013-SUPRA.zip",
                               "type": "N/A", "safe_to_convert": "⚠ MISSING, ZIP corrupted"}]

for year in all_years:
    folder = BASE_DIR / f"annee_{year}"
    all_files = [f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() != ".zip"]
    csv_files = [f for f in all_files if f.suffix.lower() == ".csv"]
    xls_files = [f for f in all_files if f.suffix.lower() == ".xls"]
    xlsx_files = [f for f in all_files if f.suffix.lower() == ".xlsx"]
    other_files = [f for f in all_files if f not in csv_files and f not in xls_files and f not in xlsx_files]

    print(f"\n{'─' * 60}")
    print(f"  YEAR {year}  |  {folder.relative_to(BASE_DIR.parent)}")
    print(f"{'─' * 60}")
    print(f"  Files: {len(all_files)} total  (CSV: {len(csv_files)}, XLS: {len(xls_files)}, XLSX: {len(xlsx_files)}, Other: {len(other_files)})")
    if other_files:
        print(f"  Other: {[f.name for f in other_files]}")

    year_results = []
    year_join_keys = set()
    data_files = sorted(csv_files + xls_files + xlsx_files)

    # For large years (2018-2021 have 42 data files + 42 meta files), only show DEP files
    dep_files = [f for f in data_files if "_DEP" in f.name.upper() and "meta_" not in f.name.lower()]
    other_data = [f for f in data_files if f not in dep_files]

    if len(data_files) > 15:
        print(f"  (Showing DEP-level files only for brevity; {len(data_files)} total files)")
        inspect_list = dep_files[:10]
    else:
        inspect_list = data_files

    for f in inspect_list:
        print(f"\n  FILE: {f.relative_to(folder)}")
        if f.suffix.lower() == ".csv":
            r = inspect_csv(f)
        elif f.suffix.lower() == ".xls":
            r = inspect_xls(f)
        else:
            r = inspect_xlsx(f)

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
        jk = r.get('join_key')
        print(f"    Join key  : {jk if jk else '⚠ not found, check manually'}")
        print(f"    Conversion: {r.get('safe_to_convert', 'N/A')}")
        print(f"    Sample ({SAMPLE_ROWS} rows):")
        for line in r.get("sample", "N/A").splitlines():
            print(f"      {line}")

        if r.get("join_key"):
            year_join_keys.add(r["join_key"])

    # Add all files to conversion report (even uninspected ones)
    for f in data_files:
        if f not in inspect_list:
            if f.suffix.lower() == ".csv":
                year_results.append({"file": f.name, "type": "CSV", "safe_to_convert": "already CSV"})
            elif f.suffix.lower() == ".xls":
                year_results.append({"file": f.name, "type": "XLS", "safe_to_convert": "pending, needs xlrd check"})
            else:
                year_results.append({"file": f.name, "type": "XLSX", "safe_to_convert": "pending, not inspected"})

    conversion_report[year] = year_results
    join_key_summary[year] = year_join_keys if year_join_keys else {"⚠ NOT FOUND"}


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY, Join key by year")
print("=" * 70)
for year in sorted(join_key_summary.keys()):
    print(f"  {year}:  {join_key_summary[year]}")

print("\n" + "=" * 70)
print("SUMMARY, CSV conversion assessment (DEP-level files only for 2018+)")
print("=" * 70)
for year in sorted(conversion_report.keys()):
    dep_results = [r for r in conversion_report[year] if "_DEP" in r["file"].upper() or "MISSING" in r.get("safe_to_convert","")]
    if not dep_results:
        dep_results = conversion_report[year][:3]
    print(f"\n  {year} (showing DEP or representative files):")
    for r in dep_results:
        status = r.get("safe_to_convert", "N/A")
        err = f"  ← {r['error']}" if "error" in r else ""
        print(f"    {r['file']:60s}  {status}{err}")

print("\n\nSetup and inspection complete. No data was merged or modified.")
print("\nACTION ITEMS:")
print("  1. Re-download 2013 data from: https://www.insee.fr/fr/statistiques/2388413")
print("  2. All 2016 XLS and 2012/2014/2015 XLS files need xlrd (installed) to read.")
print("  3. Join key is CODGEO for 2018-2021. Verify same in 2012-2017 once inspected.")
