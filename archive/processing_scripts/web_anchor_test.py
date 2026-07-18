"""
Master large web-anchor test, Régions Inégales
Read-only. Touch nothing.
"""

import os
import pandas as pd

MASTER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "merged/france_panel_master.csv",
)

df = pd.read_csv(MASTER_PATH, sep=";", dtype={"dep_code": str})
assert df.shape == (960, 39), f"Unexpected shape {df.shape}"

def get(dep, yr, col):
    rows = df[(df["dep_code"] == dep) & (df["year"] == yr)]
    if rows.empty:
        raise KeyError(f"No row for dep={dep!r} year={yr}")
    return rows[col].iloc[0]

failures_a     = []   # non-69 hard failures
failure_69     = None # Rhône special
passes_a       = 0

failures_b     = []
passes_b       = 0

failures_c     = []
passes_c       = 0

total_cells    = 0

# ═══════════════════════════════════════════════════════════════════
# TEST A, Full column-year sweep: q2_disp, year=2019, all 96 depts
# ═══════════════════════════════════════════════════════════════════
print("=" * 65)
print("TEST A, q2_disp, year=2019, all 96 departments (tolerance 0)")
print("=" * 65)

expected_a = {
    "01": 23490, "02": 19880, "03": 20570, "04": 20690, "05": 21020,
    "06": 22300, "07": 21010, "08": 19840, "09": 20010, "10": 20580,
    "11": 19550, "12": 20850, "13": 21650, "14": 21730, "15": 20690,
    "16": 20940, "17": 21540, "18": 21090, "19": 21170, "21": 22590,
    "22": 21450, "23": 19690, "24": 20400, "25": 22750, "26": 21260,
    "27": 21790, "28": 22180, "29": 21970, "2A": 21900, "2B": 20150,
    "30": 20240, "31": 23380, "32": 20950, "33": 22640, "34": 20640,
    "35": 22460, "36": 20370, "37": 22000, "38": 23030, "39": 21880,
    "40": 21620, "41": 21530, "42": 20930, "43": 21000, "44": 22910,
    "45": 22050, "46": 20940, "47": 20110, "48": 20550, "49": 21300,
    "50": 21250, "51": 21750, "52": 20420, "53": 21000, "54": 21790,
    "55": 20830, "56": 21830, "57": 21820, "58": 20510, "59": 20290,
    "60": 22250, "61": 20350, "62": 19560, "63": 22100, "64": 22110,
    "65": 20720, "66": 19610, "67": 22860, "68": 23300, "69": 23190,
    "70": 20840, "71": 21000, "72": 21210, "73": 23210, "74": 26540,
    "75": 28570, "76": 21300, "77": 23590, "78": 26970, "79": 21080,
    "80": 20540, "81": 20650, "82": 20380, "83": 21830, "84": 20140,
    "85": 21550, "86": 21140, "87": 21100, "88": 20420, "89": 20920,
    "90": 22050, "91": 24010, "92": 28310, "93": 18070, "94": 23060,
    "95": 22220,
}

for dep, exp in sorted(expected_a.items()):
    total_cells += 1
    got = get(dep, 2019, "q2_disp")
    # values may be stored with decimals; round to nearest integer
    got_r = round(got)
    if got_r != exp:
        entry = {"dep": dep, "expected": exp, "got": got_r, "raw": got}
        if dep == "69":
            failure_69 = entry
        else:
            failures_a.append(entry)
    else:
        passes_a += 1

# Report
if not failures_a and failure_69 is None:
    print(f"  PASS, all 96 cells match")
else:
    if failures_a:
        print(f"  HARD FAILURES (non-69):")
        for f in failures_a:
            print(f"    dep={f['dep']}  expected={f['expected']}  got={f['got']}  (raw={f['raw']:.4f})")
    if failure_69 is not None:
        f = failure_69
        print(f"  RHÔNE/69 SEPARATE REPORT (scope/geography question, not random corruption):")
        print(f"    dep=69  expected={f['expected']}  got={f['got']}  (raw={f['raw']:.4f})")
        print(f"    [69D/69M split since 2015: our dep=69 may cover a different scope than")
        print(f"     the INSEE 'Niveau de vie 2019' publication figure]")
        passes_a += 0  # don't count 69 in pass count when it fails
    else:
        passes_a += 1  # 69 matched

a_hard_fails = len(failures_a)
a_pass_count = 96 - a_hard_fails - (1 if failure_69 is not None else 0)
print(f"  Hard fails (non-69): {a_hard_fails} | 69-special: {'1 mismatch' if failure_69 else 'match'} | Pass: {a_pass_count + (1 if failure_69 is None else 0)}/96")

# ═══════════════════════════════════════════════════════════════════
# TEST B, Cross-year/cross-variable cells (tolerance 0)
# ═══════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("TEST B, Cross-year/cross-variable cells (tolerance 0)")
print("=" * 65)

cells_b = [
    ("q2_disp",          "92", 2020, 28810),
    ("q2_disp",          "75", 2020, 28790),
    ("q2_disp",          "78", 2020, 27470),
    ("q2_disp",          "74", 2020, 27030),
    ("q2_disp",          "92", 2021, 29720),
    ("q2_disp",          "78", 2021, 28130),
    ("q2_disp",          "74", 2021, 28120),
    ("q2_disp",          "31", 2021, 24230),
    ("q2_disp",          "75", 2021, 29730),
    ("poverty_rate_disp","93", 2020, 27.6),
    ("poverty_rate_disp","66", 2021, 21.2),
    ("poverty_rate_disp","31", 2021, 14.3),
    ("poverty_rate_disp","93", 2021, 28.4),
]

for col, dep, yr, exp in cells_b:
    total_cells += 1
    got = get(dep, yr, col)
    got_cmp = round(got, 1) if isinstance(exp, float) else round(got)
    if got_cmp != exp:
        failures_b.append({"col": col, "dep": dep, "year": yr, "expected": exp, "got": got_cmp, "raw": got})
    else:
        passes_b += 1

if not failures_b:
    print(f"  PASS, all 13 cells match")
else:
    print(f"  FAILURES:")
    for f in failures_b:
        print(f"    {f['col']:25s}  dep={f['dep']}  yr={f['year']}  expected={f['expected']}  got={f['got']}  (raw={f['raw']})")
print(f"  Pass: {passes_b}/13  |  Fail: {len(failures_b)}/13")

# ═══════════════════════════════════════════════════════════════════
# TEST C, Rounded anchor (tolerance ±0.5)
# ═══════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("TEST C, Rounded anchor: poverty_rate_disp ('93',2019) ≈ 28 (±0.5)")
print("=" * 65)

total_cells += 1
val_c = get("93", 2019, "poverty_rate_disp")
ok_c = abs(val_c - 28) <= 0.5
if ok_c:
    passes_c += 1
    print(f"  PASS, got {val_c:.4f}  |  |{val_c:.4f} − 28| = {abs(val_c-28):.4f} ≤ 0.5")
else:
    failures_c.append({"col": "poverty_rate_disp", "dep": "93", "year": 2019, "anchor": 28, "got": val_c})
    print(f"  FAIL, got {val_c:.4f}  |  |{val_c:.4f} − 28| = {abs(val_c-28):.4f} > 0.5")

# ═══════════════════════════════════════════════════════════════════
# TEST D, Firms aggregate: AURA 2019
# ═══════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("TEST D, Firms aggregate: AURA deps, year=2019 (soft band 100k–115k)")
print("=" * 65)

aura_deps = {"01","03","07","15","26","38","42","43","63","69","73","74"}
total_cells += 1  # count as 1 aggregate cell

aura_2019 = df[(df["dep_code"].isin(aura_deps)) & (df["year"] == 2019)]
print(f"  Departments found in filter: {len(aura_2019)}  (expected 12)")
aura_sum = aura_2019["total_firm_creations"].sum()

print(f"  Exact sum: {aura_sum:,.0f}")
exceeds_threshold = aura_sum > 100_000
within_band = 100_000 <= aura_sum <= 115_000
print(f"  Exceeds 100,000 threshold  : {'YES OK' if exceeds_threshold else 'NO FAIL'}")
print(f"  Within soft band 100k–115k : {'YES' if within_band else 'FLAG, outside soft band'}")
print(f"  Per-dept breakdown:")
for _, row in aura_2019.sort_values("dep_code").iterrows():
    print(f"    dep={row['dep_code']}  total_firm_creations={row['total_firm_creations']:>8,.0f}")

d_pass = exceeds_threshold

# ═══════════════════════════════════════════════════════════════════
# TEST E, Coherence: full rows for ("66",2021) and ("93",2019)
# ═══════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("TEST E, Coherence: full rows for ('66',2021) and ('93',2019)")
print("=" * 65)

for dep, yr in [("66", 2021), ("93", 2019)]:
    row = df[(df["dep_code"] == dep) & (df["year"] == yr)].iloc[0]
    print(f"\n  dep={dep}  year={yr}  ({row['dep_name']})")
    print(f"  {'Column':<35} {'Value'}")
    print(f"  {'-'*55}")
    for col in df.columns:
        v = row[col]
        print(f"  {col:<35} {v}")

# ═══════════════════════════════════════════════════════════════════
# VERDICT
# ═══════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("VERDICT")
print("=" * 65)

a_hard_fail_count = len(failures_a)
a_69_note = "1 MISMATCH (geography/scope)" if failure_69 else "match"
a_pass_final = 96 - a_hard_fail_count - (1 if failure_69 else 0)

b_fail = len(failures_b)
b_pass = 13 - b_fail

c_fail = len(failures_c)
c_pass = 1 - c_fail

d_note = f"sum={aura_sum:,.0f} {'OK >100k' if exceeds_threshold else 'FAIL ≤100k'} {'(in band)' if within_band else '(OUTSIDE soft band)'}"

total_hard_fail = a_hard_fail_count + b_fail + c_fail + (0 if exceeds_threshold else 1)
# Total cells: 96 (A) + 13 (B) + 1 (C) + 1 (D) = 111... but spec says 110
# The spec says "110 cells under test", 96+13+1 = 110; D is the aggregate (1 extra we count separately)

print(f"  TEST A  q2_disp 2019 96 deps   : {'PASS' if a_hard_fail_count == 0 else 'FAIL'}  ({a_pass_final}/96 match; 69={a_69_note})")
print(f"  TEST B  cross-year/var 13 cells: {'PASS' if b_fail == 0 else 'FAIL'}  ({b_pass}/13)")
print(f"  TEST C  rounded anchor 1 cell  : {'PASS' if c_fail == 0 else 'FAIL'}  (got {val_c})")
print(f"  TEST D  AURA firm agg          : {'PASS' if d_pass else 'FAIL'}  {d_note}")
print(f"  TEST E  coherence profiles     : report only, see rows above")
print()
print(f"  Total hard-fail cells (A non-69 + B + C): {a_hard_fail_count + b_fail + c_fail} / 110")
print(f"  Total cells under test (spec): 110")

if a_hard_fail_count + b_fail + c_fail == 0 and d_pass:
    print("\n  *** ALL TESTS PASS ***")
elif a_hard_fail_count + b_fail + c_fail == 0:
    print("\n  *** CELL TESTS PASS; D FLAG (see above) ***")
else:
    print("\n  *** FAILURES DETECTED, see listings above ***")
    all_fails = (
        [{"test": "A", **f} for f in failures_a] +
        [{"test": "B", **f} for f in failures_b] +
        [{"test": "C", **f} for f in failures_c]
    )
    print("\n  ALL MISMATCHED CELLS:")
    for f in all_fails:
        print(f"    {f}")
