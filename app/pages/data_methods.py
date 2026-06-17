import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from utils import page_header, source_card, BLUE, RED

page_header(
    "Methods",
    "Six official sources · each cross-checked before entering the model",
)

# ══════════════════════════════════════════════════════════════════════════
# 1. DATA SOURCES  (real text from DATA_SOURCES.md)
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<h3 class="ri-section-h">Data sources</h3>', unsafe_allow_html=True)

SOURCES = [
    {
        "name":       "Filosofi: Income, Poverty &amp; Inequality",
        "producer":   "INSEE · Fichier Localisé Social et Fiscal",
        "desc":       (
            "Household income deciles and quartiles, Gini coefficient, S80/S20, poverty rates "
            "at department level, 2012-2021. Two income concepts: "
            "declared (<code>_dec</code>) and disposable (<code>_disp</code>). "
            "<b>Verification:</b> 96-department sweep for median income and Gini across "
            "representative years; spot anchors against published INSEE summary tables."
        ),
    },
    {
        "name":       "SIDE: Firm Creations",
        "producer":   "INSEE · Système d'Information sur les Démographies d'Entreprises",
        "desc":       (
            "Total firm creations per department per year (2012-2021); legal-form breakdown "
            "(individual entrepreneur, SARL, SAS, other). "
            "SIDE/BURE totals run ~2-5% above harmonized INSEE Première figures "
            "(definitional difference, not a data error). "
            "<b>Verification:</b> aggregate cross-check against selected published INSEE "
            "Première annual figures."
        ),
    },
    {
        "name":       "Localised Unemployment Rate",
        "producer":   "INSEE · Estimations de taux de chômage localisés",
        "desc":       (
            "ILO quarterly unemployment rate, quarterly T1-T4 averaged to annual mean. "
            "Q4-2025 vintage used throughout. "
            "<b>Verification (2026-06-11):</b> all 960 (dept, year) pairs independently fetched "
            "from INSEE BDM SDMX API. Result: 818/960 exact (85.2%), "
            "142/960 within ±0.1 (14.8%), 0 beyond ±0.1."
        ),
    },
    {
        "name":       "Doctor Density (RPPS)",
        "producer":   "DREES · Répertoire partagé des professionnels de santé",
        "desc":       (
            "Active doctors per 100,000 inhabitants at 1 January of each year. "
            "Scope: <em>actifs occupés</em> (at least one ongoing liberal or salaried activity). "
            "Used as a quality-of-life proxy, not a direct healthcare predictor. "
            "<b>Verification (2026-06-13):</b> density recomputed from headcounts and population "
            "for all 960 cells. Max relative deviation 0.0025%."
        ),
    },
    {
        "name":       "Education: Higher-Ed Share",
        "producer":   "INSEE · Diplômes et formation (RP census snapshots 2011, 2016, 2022)",
        "desc":       (
            "Share of non-schooled population aged 15+ with any higher-education diploma "
            "(Bac+2 and above). Three census snapshots linearly interpolated to fill the full "
            "2012-2021 annual panel; only 2016 is a real within-panel observation. "
            "<b>Verification:</b> 288/288 dept re-derivation across all three anchor years; "
            "formula confirmed against ANCT indicator <code>p_diplsup15</code>."
        ),
    },
    {
        "name":       "Urban/Rural Density",
        "producer":   "INSEE · Grille de densité 2025 (RP2021-based)",
        "desc":       (
            "Percent urban by department: <code>pct_urban = 100 - P_RURAL</code>. "
            "Time-invariant; one RP2021 classification per department replicated across all years. "
            "Uses the Grille de densité (1 km² grid), not the unités urbaines framework. "
            "Range: 10.3% (Creuse) to 100% (Paris). "
            "<b>Verification:</b> range and ranking cross-checked against published INSEE typology."
        ),
    },
]

for src in SOURCES:
    st.markdown(source_card(src["name"], src["producer"], src["desc"]), unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# 2. LIMITATIONS  (real text from FINDINGS.md)
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<h3 class="ri-section-h">Limitations</h3>', unsafe_allow_html=True)

LIMITATIONS = [
    (
        "Registrations, not survival.",
        "SIDE counts legal registrations including auto-entrepreneurs who may cease activity quickly. "
        "The model captures entry propensity, not sustained entrepreneurial activity.",
    ),
    (
        "Mostly a cross-sectional story.",
        "Roughly 70% of the predictive variance is between departments rather than within them over time. "
        "Results describe which kinds of departments produce more entrepreneurs, not why creation rates "
        "rose or fell in a given year.",
    ),
    (
        "2016–2018 SIDE measurement artefact.",
        "INSEE reformed the registration system in this period (auto-entrepreneur counting rules changed), "
        "causing a structural break in raw firm-creation counts. Year fixed effects in LOYO partly absorb "
        "this, but residual inflation in those years cannot be ruled out.",
    ),
    (
        "Correlational, not causal.",
        "No instrumental variable or quasi-experimental design is applied. The model shows which "
        "departmental characteristics predict firm-creation rates, not what would change if those "
        "characteristics changed.",
    ),
    (
        "Education interpolated.",
        "Higher-ed share is observed at three census points (2011, 2016, 2022) and linearly interpolated "
        "for all other years. Year-to-year variation in this variable is artificial by construction.",
    ),
    (
        "Doctor density as amenity proxy.",
        "Physician density is used as a quality-of-life proxy for the opportunity environment. Its positive "
        "contribution likely reflects broader urban amenity endowments rather than a direct healthcare "
        "mechanism.",
    ),
]

items = "".join(
    f'<div class="ri-lim-item">'
    f'<span class="ri-lim-label">{i+1}. {title}</span> {body}'
    f"</div>"
    for i, (title, body) in enumerate(LIMITATIONS)
)
st.markdown(items, unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# 3. PREPRINT BLOCK
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<h3 class="ri-section-h">Publication</h3>', unsafe_allow_html=True)

st.markdown(
    """
<div class="ri-preprint">
    <strong>Preprint forthcoming</strong><br>
    <span style="color:#888;font-size:0.88rem">
        Working paper will be posted to SSRN / EconPapers.<br>
        Link: <em>TODO — add preprint URL when posted</em>
    </span>
    <br><br>
    <strong>Citation (placeholder)</strong><br>
    <span style="font-family:monospace;font-size:0.82rem;color:#444">
        Author (2026). <em>Régions Inégales: Opportunity vs. Necessity Entrepreneurship
        Across French Departments, 2012–2021.</em> Working paper. TODO-REAL: add DOI.
    </span>
    <br><br>
    <strong>Data & replication</strong><br>
    <span style="color:#888;font-size:0.88rem">
        All six source files are publicly available from INSEE and DREES. See source cards above for download URLs.
        Replication code: TODO-REAL: link to GitHub repository.
    </span>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)
st.caption(
    f"Panel: 960 observations · 96 metropolitan departments × 10 years (2012–2021) · "
    f"Model: XGBoost + SHAP · Validation: LODO R² = 0.674"
)
