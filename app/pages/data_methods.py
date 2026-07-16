import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from utils import page_header, source_card, render_footer, BLUE, RED

page_header(
    "Methods",
    "Nine official sources · each cross-checked before entering the model",
)

# ══════════════════════════════════════════════════════════════════════════
# 1. DATA SOURCES  (real text from DATA_SOURCES.md)
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<h3 class="ri-section-h">Data sources</h3>', unsafe_allow_html=True)

_FILOSOFI_YEAR_LINKS = {
    2012: "https://www.insee.fr/fr/statistiques/2043745",
    2013: "https://www.insee.fr/fr/statistiques/2388413",
    2014: "https://www.insee.fr/fr/statistiques/3126151",
    2015: "https://www.insee.fr/fr/statistiques/3560118",
    2016: "https://www.insee.fr/fr/statistiques/4190006",
    2017: "https://www.insee.fr/fr/statistiques/4291712",
    2018: "https://www.insee.fr/fr/statistiques/5009218",
    2019: "https://www.insee.fr/fr/statistiques/6036907",
    2020: "https://www.insee.fr/fr/statistiques/6692220",
    2021: "https://www.insee.fr/fr/statistiques/7756855",
}

SOURCES = [
    {
        "name":       "Filosofi: Income, Poverty &amp; Inequality",
        "url":        None,
        "year_links": _FILOSOFI_YEAR_LINKS,
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
        "url":        "https://www.insee.fr/fr/statistiques/8557644",
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
        "url":        "https://www.insee.fr/fr/statistiques/2012804",
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
        "url":        "https://data.drees.solidarites-sante.gouv.fr/explore/dataset/la-demographie-des-professionnels-de-sante-depuis-2012/information/",
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
        "url":        "https://www.insee.fr/fr/statistiques/8581488",
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
        "url":        "https://www.insee.fr/fr/information/8571524",
        "producer":   "INSEE · Grille de densité 2025 (RP2021-based)",
        "desc":       (
            "Percent urban by department: <code>pct_urban = 100 - P_RURAL</code>. "
            "Time-invariant; one RP2021 classification per department replicated across all years. "
            "Uses the Grille de densité (1 km² grid), not the unités urbaines framework. "
            "Range: 10.3% (Creuse) to 100% (Paris). "
            "<b>Verification:</b> range and ranking cross-checked against published INSEE typology."
        ),
    },
    {
        "name":       "Live Births (DS_NAISSANCES_FECONDITE_SERIES)",
        "url":        "https://www.data.gouv.fr/datasets/naissances-et-fecondite-series-longues",
        "producer":   "INSEE · Naissances et fécondité — séries longues",
        "desc":       (
            "Live births at place of residence (<code>LVB_PLACE_RES</code>, all ages, <code>AGE=_T</code>), "
            "2012–2021, metropolitan departments only. "
            "Residence attribution preferred over place of registration (<code>LVB_PLACE_REG</code>) for a "
            "panel of departmental characteristics. "
            "Panel columns: <code>live_births</code> (raw count) and <code>birth_rate</code> "
            "(live births per 1,000 inhabitants, computed at merge time using population denominator). "
            "<b>Verification:</b> 960/960 (dep, year) pairs; 0 nulls; "
            "rate range 6.5–18.8 per 1,000, consistent with published INSEE natality profiles."
        ),
    },
    {
        "name":       "Deaths (DS_ETAT_CIVIL_DECES_COMMUNES)",
        "url":        "https://www.data.gouv.fr/datasets/deces-et-mortalite-series-longues/",
        "producer":   "INSEE · Décès et mortalité — séries longues",
        "desc":       (
            "Deaths by department of occurrence (<code>EC_MEASURE=DTH</code>, <code>OBS_STATUS=A</code> only), "
            "2012–2021, metropolitan departments only. "
            "<code>OBS_STATUS=M</code> (missing/not applicable) rows dropped throughout. "
            "Panel columns: <code>deaths</code> (raw count) and <code>death_rate</code> "
            "(deaths per 1,000 inhabitants, computed at merge time). "
            "<b>Verification:</b> 960/960 (dep, year) pairs; 0 nulls; "
            "rate range 5.3–17.4 per 1,000, consistent with published INSEE mortality profiles."
        ),
    },
    {
        "name":       "Marriages (INSEE État civil DEP6 annual files)",
        "url":        None,
        "year_links": {
            2012: "https://www.insee.fr/fr/statistiques/2020625",
            2013: "https://www.insee.fr/fr/statistiques/2020490",
            2014: "https://www.insee.fr/fr/statistiques/1913527",
            2015: "https://www.insee.fr/fr/statistiques/2561535",
            2016: "https://www.insee.fr/fr/statistiques/3317603",
            2017: "https://www.insee.fr/fr/statistiques/3704307",
            2018: "https://www.insee.fr/fr/statistiques/4273672",
            2019: "https://www.insee.fr/fr/statistiques/5012966",
            2020: "https://www.insee.fr/fr/statistiques/6045407",
            2021: "https://www.insee.fr/fr/statistiques/6790710",
        },
        "producer":   "INSEE · État civil, DEP6 (mariages par département et région)",
        "desc":       (
            "Total marriages (HF + same-sex HH + FF) by department of marriage, 2012–2021. "
            "Three file formats across years: single-sheet XLS (2012–2013), multi-sheet XLS/XLSX "
            "with HF/HH/FF sheets (2014–2017, 2020), and CSV with REGDEP_MAR codes (2018, 2019, 2021). "
            "<b>Critical:</b> CSV files count married <em>persons</em>, not weddings (2 per wedding) — "
            "NBMAR divided by 2 to harmonise with XLS files. "
            "Geography of marriage (not residence). "
            "Panel columns: <code>marriages</code> (weddings) and <code>marriage_rate</code> "
            "(per 1,000 inhabitants). "
            "<b>Verification:</b> 960/960 (dep, year) pairs; national totals consistent with INSEE published "
            "figures (239,840 in 2012; 150,545 in 2020 COVID drop; ~213,000 recovery in 2021); "
            "rate range 1.8–4.5 per 1,000."
        ),
    },
]

_LINK = (
    '<a href="{url}" target="_blank" rel="noopener noreferrer" '
    'style="color:inherit;text-decoration:underline;text-decoration-color:#aaa;">{name}</a>'
)

def _year_btns(year_links: dict) -> str:
    return "".join(
        f'<a href="{u}" target="_blank" rel="noopener noreferrer" class="ri-year-btn">{y}</a>'
        for y, u in year_links.items()
    )

def _src_btn(url: str) -> str:
    return f'<a href="{url}" target="_blank" rel="noopener noreferrer" class="ri-src-btn">Open source</a>'

for src in SOURCES:
    url = src.get("url") or ""
    year_links = src.get("year_links", {})
    name = _LINK.format(url=url, name=src["name"]) if url else src["name"]
    if year_links:
        links_html = _year_btns(year_links)
    elif url:
        links_html = _src_btn(url)
    else:
        links_html = ""
    st.markdown(source_card(name, src["producer"], src["desc"], links_html), unsafe_allow_html=True)

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
        "this, but some residual inflation in those years likely remains.",
    ),
    (
        "Correlational, not causal.",
        "No instrumental variable or quasi-experimental design is applied. The model shows which "
        "departmental characteristics predict firm-creation rates, not what would change if those "
        "characteristics changed.",
    ),
    (
        "Education interpolated against a post-panel anchor.",
        "Higher-ed share is observed at census snapshots (2011, 2016, 2022) and linearly interpolated. "
        "The 2022 anchor lies outside the panel window, so 2017–2021 interpolated values embed future "
        "information. The LOYO 2021 fold is mildly contaminated; year-to-year variation in this variable "
        "is artificial by construction.",
    ),
    (
        "pct_urban is a single vintage classification applied to all years.",
        "The density classification (Grille de densité, RP2021/2025 vintage) is time-invariant and applied "
        "uniformly across 2012–2021. It is a forward look-ahead for early years and contributes only "
        "cross-sectional signal.",
    ),
    (
        "pct_wages uses a different income concept from the other income variables.",
        "pct_wages is derived from the DEC income concept, while q2_disp, gini_disp, and poverty_rate_disp "
        "use the DISP (disposable income) concept. Cross-concept comparisons within the feature matrix "
        "should be treated with caution.",
    ),
    (
        "Doctor density as amenity proxy.",
        "Physician density is used as a quality-of-life proxy for the opportunity environment. Its positive "
        "contribution captures broader urban amenity endowments, not a direct healthcare effect.",
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
    <strong>Preprint</strong><br>
    <em>Evidence Against the Necessity Hypothesis of Entrepreneurship: A Panel Machine-Learning Analysis of French Departments, 2012&ndash;2021</em><br>
    <span style="font-size:0.88rem">
        Zenodo: #TODO &nbsp;|&nbsp; DOI: #TODO
    </span>
    <br><br>
    <strong>Citation</strong><br>
    <span style="font-family:monospace;font-size:0.82rem;color:#444">
        #TODO
    </span>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<h3 class="ri-section-h">Project links</h3>', unsafe_allow_html=True)

st.markdown(
    """
<div class="ri-preprint">
    <strong>Live app</strong><br>
    <span style="font-size:0.88rem">
        <a href="https://regions-inegales.streamlit.app/" target="_blank"
           style="color:#0055A4;">https://regions-inegales.streamlit.app/</a>
    </span>
    <br><br>
    <strong>Source code</strong><br>
    <span style="font-size:0.88rem">
        <a href="https://github.com/ShailKPatel/Regions-Inegales" target="_blank"
           style="color:#0055A4;">https://github.com/ShailKPatel/Regions-Inegales</a>
    </span>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)
st.caption(
    f"Panel: 960 observations · 96 metropolitan departments × 10 years (2012–2021) · "
    f"Model: XGBoost + SHAP · Validation: LODO R² = 0.678"
)

render_footer()
