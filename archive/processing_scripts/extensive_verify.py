#!/usr/bin/env python3
"""
Extensive re-verification: Layer 1 (structural) + Layer 2 (raw spot-check).
Layer 3 (web anchors) run separately via WebFetch.
"""

import os, random, zipfile
import pandas as pd
import numpy as np

PROJECT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILO_PATH = os.path.join(PROJECT, "Base niveau administratif/filosofi_income_panel.csv")
FIRMS_PATH = os.path.join(PROJECT, "firms_clean/firms_panel.csv")
ZIP_PATH  = os.path.join(PROJECT, "d'entreprises/DS_SIDE_CREA_DEP_REG_NAT_2024_CSV_FR.zip")
DATA_FILE = "DS_SIDE_CREA_DEP_REG_NAT_2024_data.csv"

METRO_DEPS = (
    {f"{i:02d}" for i in range(1, 20)}
    | {f"{i:02d}" for i in range(21, 96)}
    | {"2A", "2B"}
)

SEP = "=" * 68

def section(title):
    print(f"\n{SEP}\n{title}\n{SEP}")

violations = {"filo_l1": 0, "firms_l1": 0}

# ─── Load panels ──────────────────────────────────────────────────────────────
filo  = pd.read_csv(FILO_PATH,  sep=";", dtype={"dep_code": str})
firms = pd.read_csv(FIRMS_PATH, sep=";", dtype={"dep_code": str})
filo["year"]  = filo["year"].astype(int)
firms["year"] = firms["year"].astype(int)

print(f"Filosofi shape : {filo.shape}")
print(f"Firms shape    : {firms.shape}")

# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 1-A, FILOSOFI STRUCTURAL INTEGRITY
# ══════════════════════════════════════════════════════════════════════════════
section("LAYER 1-A  Filosofi structural integrity (960 rows)")

# ── Check 1: decile ordering d1 < q1 < q2 < q3 < d9 ─────────────────────────
print("\n[F1] Decile ordering: d1_disp < q1_disp < q2_disp < q3_disp < d9_disp")
bad = filo[
    ~(
        (filo.d1_disp < filo.q1_disp) &
        (filo.q1_disp < filo.q2_disp) &
        (filo.q2_disp < filo.q3_disp) &
        (filo.q3_disp < filo.d9_disp)
    )
]
if bad.empty:
    print("  PASS, all 960 rows satisfy strict ordering OK")
else:
    violations["filo_l1"] += len(bad)
    print(f"  FAIL, {len(bad)} violation(s):")
    print(bad[["dep_code","year","d1_disp","q1_disp","q2_disp","q3_disp","d9_disp"]].to_string(index=False))

# Same check for dec (declared income) columns
bad_dec = filo[
    ~(
        (filo.d1_dec < filo.q1_dec) &
        (filo.q1_dec < filo.q2_dec) &
        (filo.q2_dec < filo.q3_dec) &
        (filo.q3_dec < filo.d9_dec)
    )
]
if bad_dec.empty:
    print("  PASS, dec-income ordering also holds OK")
else:
    violations["filo_l1"] += len(bad_dec)
    print(f"  FAIL (dec), {len(bad_dec)} violation(s):")
    print(bad_dec[["dep_code","year","d1_dec","q1_dec","q2_dec","q3_dec","d9_dec"]].to_string(index=False))

# ── Check 2: Gini bounds ──────────────────────────────────────────────────────
print("\n[F2] Gini bounds: hard error outside (0,1); soft flag outside (0.15,0.55)")
for col in ["gini_disp", "gini_dec"]:
    hard = filo[(filo[col] <= 0) | (filo[col] >= 1)]
    soft = filo[(filo[col] < 0.15) | (filo[col] > 0.55)]
    if hard.empty:
        print(f"  {col}: no hard errors OK")
    else:
        violations["filo_l1"] += len(hard)
        print(f"  {col}: HARD ERROR, {len(hard)} rows outside (0,1)")
        print(hard[["dep_code","year",col]].to_string(index=False))
    if not soft.empty:
        print(f"  {col}: SOFT FLAG, {len(soft)} rows outside (0.15,0.55):")
        print(soft[["dep_code","year",col]].to_string(index=False))
    else:
        print(f"  {col}: soft range fine OK")

# ── Check 3: D9/D1 ratio consistency ─────────────────────────────────────────
print("\n[F3] D9/D1 ratio consistency: recomputed d9_disp/d1_disp vs stored d9_d1_disp (tolerance 1%)")
filo["_ratio_check"] = filo["d9_disp"] / filo["d1_disp"]
filo["_ratio_pct_err"] = (filo["_ratio_check"] - filo["d9_d1_disp"]).abs() / filo["d9_d1_disp"] * 100
bad_ratio = filo[filo["_ratio_pct_err"] > 1.0]
if bad_ratio.empty:
    print("  PASS, all 960 ratios match within 1% OK")
    print(f"  Max discrepancy: {filo['_ratio_pct_err'].max():.4f}%")
else:
    violations["filo_l1"] += len(bad_ratio)
    print(f"  FAIL, {len(bad_ratio)} mismatches (>1%):")
    print(bad_ratio[["dep_code","year","d9_disp","d1_disp","_ratio_check","d9_d1_disp","_ratio_pct_err"]].to_string(index=False))
filo.drop(columns=["_ratio_check","_ratio_pct_err"], inplace=True)

# Same for dec columns
filo["_ratio_check"] = filo["d9_dec"] / filo["d1_dec"]
filo["_ratio_pct_err"] = (filo["_ratio_check"] - filo["d9_d1_dec"]).abs() / filo["d9_d1_dec"] * 100
bad_ratio_dec = filo[filo["_ratio_pct_err"] > 1.0]
if bad_ratio_dec.empty:
    print(f"  PASS (dec cols), max discrepancy: {filo['_ratio_pct_err'].max():.4f}% OK")
else:
    violations["filo_l1"] += len(bad_ratio_dec)
    print(f"  FAIL (dec), {len(bad_ratio_dec)} mismatches:")
    print(bad_ratio_dec[["dep_code","year","d9_d1_dec","_ratio_pct_err"]].to_string(index=False))
filo.drop(columns=["_ratio_check","_ratio_pct_err"], inplace=True)

# ── Check 4: Poverty rate bounds ──────────────────────────────────────────────
print("\n[F4] Poverty rate bounds: 0 < poverty_rate_disp < 60")
for col in ["poverty_rate_disp", "poverty_rate_dec"]:
    bad = filo[(filo[col] <= 0) | (filo[col] >= 60)]
    if bad.empty:
        print(f"  {col}: PASS, all in (0,60) OK  [range: {filo[col].min():.1f}–{filo[col].max():.1f}]")
    else:
        violations["filo_l1"] += len(bad)
        print(f"  {col}: FAIL, {len(bad)} rows outside (0,60):")
        print(bad[["dep_code","year",col]].to_string(index=False))

# ── Check 5: Income composition sums ─────────────────────────────────────────
print("\n[F5] Income composition: pct_wages + pct_unemployment + pct_pensions + pct_capital_gains + pct_other")
pct_cols = ["pct_wages","pct_unemployment","pct_capital_gains","pct_pensions","pct_other"]
filo["_pct_sum"] = filo[pct_cols].sum(axis=1)
print(f"  Sum range: {filo['_pct_sum'].min():.1f} – {filo['_pct_sum'].max():.1f}")
print(f"  Mean sum: {filo['_pct_sum'].mean():.2f}  Std: {filo['_pct_sum'].std():.3f}")
outside = filo[(filo["_pct_sum"] < 90) | (filo["_pct_sum"] > 110)]
if outside.empty:
    print(f"  PASS, all rows sum within 90–110 OK")
else:
    print(f"  FLAG, {len(outside)} rows outside 90–110 (may indicate partial coverage):")
    print(outside[["dep_code","year","_pct_sum"] + pct_cols].head(10).to_string(index=False))
filo.drop(columns=["_pct_sum"], inplace=True)

# ── Check 6: Year-over-year stability per department ─────────────────────────
print("\n[F6] YoY stability: flag |Δ%| > 25% in q2_disp per department")
filo_sorted = filo.sort_values(["dep_code","year"])
filo_sorted["_q2_pct_chg"] = filo_sorted.groupby("dep_code")["q2_disp"].pct_change() * 100
jumps = filo_sorted[filo_sorted["_q2_pct_chg"].abs() > 25].copy()
if jumps.empty:
    max_jump = filo_sorted["_q2_pct_chg"].abs().max()
    print(f"  PASS, no year-over-year jump > 25% OK  (max observed: {max_jump:.2f}%)")
else:
    violations["filo_l1"] += len(jumps)
    print(f"  FAIL, {len(jumps)} jumps > 25%:")
    print(jumps[["dep_code","dep_name","year","q2_disp","_q2_pct_chg"]].to_string(index=False))

print(f"\n{'─'*40}")
print(f"LAYER 1-A FILOSOFI TOTAL VIOLATIONS: {violations['filo_l1']}")
print(f"{'─'*40}")

# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 1-B, FIRMS STRUCTURAL INTEGRITY
# ══════════════════════════════════════════════════════════════════════════════
section("LAYER 1-B  Firms structural integrity (960 rows)")

# ── Check 1: Non-negativity ───────────────────────────────────────────────────
print("\n[FR1] Non-negativity: all creation counts >= 0")
count_cols = [c for c in firms.columns if c.startswith("creations_") or c == "total_firm_creations"]
neg = firms[(firms[count_cols] < 0).any(axis=1)]
if neg.empty:
    print(f"  PASS, no negative values in any of {len(count_cols)} columns OK")
    print(f"  Minimum value across all count columns: {firms[count_cols].min().min():.0f}")
else:
    violations["firms_l1"] += len(neg)
    print(f"  FAIL, {len(neg)} rows contain negative values:")
    print(neg.to_string(index=False))

# ── Check 2: Legal-form component sum vs total ────────────────────────────────
print("\n[FR2] Legal-form components sum vs total_firm_creations")
lf_cols = ["creations_individual","creations_sarl","creations_sas","creations_other_legal"]
firms["_lf_sum"] = firms[lf_cols].sum(axis=1)
firms["_lf_gap"] = firms["total_firm_creations"] - firms["_lf_sum"]
firms["_lf_pct"]  = firms["_lf_gap"] / firms["total_firm_creations"] * 100
print(f"  Gap (total − lf_sum): min={firms['_lf_gap'].min():.0f}  max={firms['_lf_gap'].max():.0f}  mean={firms['_lf_gap'].mean():.2f}")
print(f"  Gap as % of total:    min={firms['_lf_pct'].min():.3f}%  max={firms['_lf_pct'].max():.3f}%  mean={firms['_lf_pct'].mean():.3f}%")
large = firms[firms["_lf_gap"].abs() > firms["total_firm_creations"] * 0.02]
if large.empty:
    print("  PASS, legal-form components sum within 2% of total for all rows OK")
else:
    violations["firms_l1"] += len(large)
    print(f"  FLAG, {len(large)} rows with >2% gap:")
    print(large[["dep_code","year","total_firm_creations","_lf_sum","_lf_gap"]].head(10).to_string(index=False))
firms.drop(columns=["_lf_sum","_lf_gap","_lf_pct"], inplace=True)

# ── Check 3: Sector component sum vs total ────────────────────────────────────
print("\n[FR3] Sector components sum vs total_firm_creations")
sec_cols = ["creations_sector_industry","creations_sector_construction","creations_sector_trade","creations_sector_services"]
firms["_sec_sum"] = firms[sec_cols].sum(axis=1)
firms["_sec_gap"] = firms["total_firm_creations"] - firms["_sec_sum"]
firms["_sec_pct"] = firms["_sec_gap"] / firms["total_firm_creations"] * 100
print(f"  Gap (total − sec_sum): min={firms['_sec_gap'].min():.0f}  max={firms['_sec_gap'].max():.0f}  mean={firms['_sec_gap'].mean():.2f}")
print(f"  Gap as % of total:     min={firms['_sec_pct'].min():.3f}%  max={firms['_sec_pct'].max():.3f}%  mean={firms['_sec_pct'].mean():.3f}%")
# Sample worst offenders
worst_sec = firms.nlargest(5, "_sec_gap")[["dep_code","year","total_firm_creations","_sec_sum","_sec_gap","_sec_pct"]]
print("  Largest sector gaps (top 5 by absolute gap):")
print(worst_sec.to_string(index=False))
large_sec = firms[firms["_sec_gap"].abs() > firms["total_firm_creations"] * 0.02]
if large_sec.empty:
    print("  PASS, sector components sum within 2% of total for all rows OK")
else:
    violations["firms_l1"] += len(large_sec)
    print(f"  FLAG, {len(large_sec)} rows with >2% gap (may reflect agriculture excluded from A21)")
firms.drop(columns=["_sec_sum","_sec_gap","_sec_pct"], inplace=True)

# ── Check 4: YoY stability ────────────────────────────────────────────────────
print("\n[FR4] YoY stability: flag |Δ%| > 60% in total_firm_creations per department")
firms_sorted = firms.sort_values(["dep_code","year"])
firms_sorted["_fc_pct_chg"] = firms_sorted.groupby("dep_code")["total_firm_creations"].pct_change() * 100
jumps_f = firms_sorted[firms_sorted["_fc_pct_chg"].abs() > 60].copy()
if jumps_f.empty:
    max_j = firms_sorted["_fc_pct_chg"].abs().max()
    print(f"  PASS, no jump > 60% OK  (max observed: {max_j:.1f}%)")
else:
    print(f"  NOTE, {len(jumps_f)} jumps > 60% (checking whether these are 2021 COVID bounce or alignment errors):")
    print(jumps_f[["dep_code","year","total_firm_creations","_fc_pct_chg"]].to_string(index=False))

# ── Check 5: Rank stability for dep 75 and dep 48 ────────────────────────────
print("\n[FR5] Rank stability: Paris (75) expected top-5, Lozère (48) expected bottom-10 every year")
rank_table = []
for yr in sorted(firms["year"].unique()):
    yr_df = firms[firms["year"]==yr].sort_values("total_firm_creations", ascending=False).reset_index(drop=True)
    yr_df["rank"] = yr_df.index + 1
    rank_75 = yr_df[yr_df.dep_code=="75"]["rank"].values[0]
    rank_48 = yr_df[yr_df.dep_code=="48"]["rank"].values[0]
    total_depts = len(yr_df)
    flag75 = " *** ANOMALY ***" if rank_75 > 5  else ""
    flag48 = " *** ANOMALY ***" if rank_48 < (total_depts - 9) else ""
    rank_table.append({
        "year": yr,
        "rank_75_Paris": rank_75, "flag75": flag75,
        "rank_48_Lozère": rank_48, "flag48": flag48,
        "total_depts": total_depts
    })

print(f"  {'year':>4} | {'rank Paris(75)':>14} | {'rank Lozère(48)':>15}")
print(f"  {'':->4}-+-{'':->14}-+-{'':->15}")
for r in rank_table:
    print(f"  {r['year']:>4} | {r['rank_75_Paris']:>14}{r['flag75']} | {r['rank_48_Lozère']:>15}{r['flag48']}")

anom = [r for r in rank_table if r["flag75"] or r["flag48"]]
if not anom:
    print("  PASS, Paris always top-5, Lozère always bottom-10 across all 10 years OK")
else:
    violations["firms_l1"] += len(anom)
    print(f"  FAIL, {len(anom)} year(s) with rank anomalies (see *** above)")

print(f"\n{'─'*40}")
print(f"LAYER 1-B FIRMS TOTAL VIOLATIONS: {violations['firms_l1']}")
print(f"{'─'*40}")

# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 2-A, FILOSOFI 15 random cells
# ══════════════════════════════════════════════════════════════════════════════
section("LAYER 2-A  Filosofi, 15 random cells (seed=123)")

print("""
NOTE: Raw Filosofi per-year source files were deleted after previous internal
verification (30/30 cells matched). No re-download was requested here.
Layer 1 structural checks substitute for raw re-verification.
Displaying 15 cells for visual plausibility inspection.
""")

random.seed(123)
filo_combos = list(zip(filo["dep_code"], filo["year"]))
filo_sample = random.sample(filo_combos, 15)

print(f"  {'dep':>4} {'dept name':<22} {'year':>4} | {'q2_disp':>8} | {'gini_disp':>9} | {'pov_rate%':>9}")
print(f"  {'':->4} {'':->22} {'':->4}-+-{'':->8}-+-{'':->9}-+-{'':->9}")
for dep, yr in sorted(filo_sample, key=lambda x: (x[0], x[1])):
    row = filo[(filo.dep_code==dep) & (filo.year==yr)].iloc[0]
    print(f"  {dep:>4} {row['dep_name']:<22} {yr:>4} | {row['q2_disp']:>8,.0f} | {row['gini_disp']:>9.3f} | {row['poverty_rate_disp']:>9.1f}")

# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 2-B, FIRMS 15 random cells vs raw ZIP
# ══════════════════════════════════════════════════════════════════════════════
section("LAYER 2-B  Firms, 15 random cells vs raw ZIP (seed=123)")

random.seed(123)
firms_combos = list(zip(firms["dep_code"], firms["year"]))
firms_sample = random.sample(firms_combos, 15)
firms_sample_set  = set(firms_sample)
sampled_deps  = {dep for dep,_ in firms_sample}
sampled_years = {yr  for _,yr  in firms_sample}

print(f"Streaming {ZIP_PATH.split('/')[-1]} to recompute 15 cells...")
raw_totals = {}
total_rows = 0
with zipfile.ZipFile(ZIP_PATH) as zf:
    with zf.open(DATA_FILE) as fh:
        for i, chunk in enumerate(pd.read_csv(
            fh, sep=";", encoding="utf-8", chunksize=1_000_000,
            dtype={"GEO":str,"GEO_OBJECT":str,"ACTIVITY":str,
                   "LEGAL_FORM":str,"SIDE_MEASURE":str,"TIME_PERIOD":str}
        )):
            total_rows += len(chunk)
            if i % 5 == 0:
                print(f"  {total_rows:,} rows scanned...")
            c = chunk[
                (chunk["SIDE_MEASURE"] == "BURE") &
                (chunk["GEO_OBJECT"]   == "DEP")  &
                (chunk["ACTIVITY"]     == "_T")   &
                (chunk["LEGAL_FORM"]   == "_T")   &
                (chunk["GEO"].isin(sampled_deps)) &
                (chunk["TIME_PERIOD"].astype(int).isin(sampled_years))
            ].copy()
            if len(c) == 0:
                continue
            c["OBS_VALUE"]   = pd.to_numeric(c["OBS_VALUE"], errors="coerce").fillna(0)
            c["TIME_PERIOD"] = c["TIME_PERIOD"].astype(int)
            for _, row in c.iterrows():
                key = (row["GEO"], row["TIME_PERIOD"])
                if key in firms_sample_set:
                    raw_totals[key] = raw_totals.get(key, 0) + row["OBS_VALUE"]

print(f"\nTotal raw rows scanned: {total_rows:,}")

matched = 0
print(f"\n  {'dep':>4} | {'year':>4} | {'panel':>8} | {'raw':>8} | match?")
print(f"  {'':->4}-+-{'':->4}-+-{'':->8}-+-{'':->8}-+-{'':->6}")
for dep, yr in sorted(firms_sample, key=lambda x: (x[0], x[1])):
    row = firms[(firms.dep_code==dep) & (firms.year==yr)].iloc[0]
    panel_val = int(row["total_firm_creations"])
    raw_val   = int(raw_totals.get((dep,yr), -1))
    ok = panel_val == raw_val
    if ok:
        matched += 1
    flag = "OK" if ok else "FAIL MISMATCH"
    print(f"  {dep:>4} | {yr:>4} | {panel_val:>8,} | {raw_val:>8,} | {flag}")

print(f"\nResult: {matched}/15 cells matched raw recompute")
if matched < 15:
    violations["firms_l1"] += (15 - matched)

# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY of Layer 1 + 2
# ══════════════════════════════════════════════════════════════════════════════
section("LAYER 1+2 SUMMARY")
print(f"  Filosofi structural violations : {violations['filo_l1']}")
print(f"  Firms structural violations    : {violations['firms_l1']}")
print(f"  Firms raw spot-check           : {matched}/15 matched")
print()
if violations["filo_l1"] == 0 and violations["firms_l1"] == 0 and matched == 15:
    print("  PRELIMINARY VERDICT: OK No structural issues found.")
    print("  Awaiting Layer 3 web anchors for absolute-value confirmation.")
else:
    print("  WARNING ISSUES FOUND, see details above before proceeding to analysis.")
