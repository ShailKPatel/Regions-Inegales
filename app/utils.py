import streamlit as st

BLUE  = "#0055A4"
RED   = "#EF4135"
WHITE = "#FFFFFF"
LIGHT_BG = "#DCDFE6"

GROUP_COLORS = {
    "Opportunity": BLUE,
    "Other":       WHITE,
    "Necessity":   RED,
}

_CSS = """
<style>
/* ── Page top margin ─────────────────────────────── */
[data-testid="stAppViewBlockContainer"],
[data-testid="stMainBlockContainer"],
.block-container {
    padding-top: 2.5rem !important;
}
/* ── Hero header (overview) ──────────────────────── */
.ri-hero-title {
    font-size: 7.6rem;
    font-weight: 900;
    color: #111111;
    margin: 0 0 0.15rem 0;
    letter-spacing: -1.5px;
    line-height: 1;
}
.ri-hero-subtitle {
    font-size: 0.92rem;
    color: #777;
    margin: 0;
    font-weight: 400;
}
/* ── Page header (subpages) ──────────────────────── */
.ri-page-title {
    font-size: 7.6rem;
    font-weight: 900;
    color: #111111;
    margin-bottom: 0.15rem;
    letter-spacing: -1.5px;
    line-height: 1;
}
.ri-page-subtitle {
    font-size: 0.92rem;
    color: #777;
    margin: 0;
}
.ri-title-rule {
    height: 4px;
    background: linear-gradient(to right, #0055A4 33.3%, #FFFFFF 33.3% 66.6%, #EF4135 66.6%);
    border: none;
    margin: 0.6rem 0 1.8rem 0;
}
/* ── Section headings ────────────────────────────── */
.ri-section-h {
    font-size: 1.05rem;
    font-weight: 700;
    color: #111111;
    margin: 1.6rem 0 0.1rem 0;
}
.ri-caption {
    font-size: 0.8rem;
    color: #999;
    margin: 0 0 0.8rem 0;
}
/* ── Stats strip ─────────────────────────────────── */
.ri-stats-strip {
    display: flex;
    gap: 0;
    margin: 0.5rem 0 1rem 0;
    border: 1px solid #C8CAD4;
    border-radius: 6px;
    overflow: visible;
}
.ri-stat-item {
    flex: 1;
    padding: 1.1rem 1.3rem;
    background: #ECEEF4;
    border-right: 1px solid #C8CAD4;
}
.ri-stat-item:first-child { border-radius: 6px 0 0 6px; }
.ri-stat-item:last-child  { border-right: none; border-radius: 0 6px 6px 0; }
.ri-stat-n {
    font-size: 2.1rem;
    font-weight: 800;
    color: #111111;
    line-height: 1;
    letter-spacing: -1px;
}
.ri-stat-n.red { color: #EF4135; }
.ri-stat-l {
    font-size: 0.73rem;
    color: #888;
    margin-top: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 500;
}
/* ── Note box ────────────────────────────────────── */
.ri-note {
    background: #E4E6EE;
    border-left: 3px solid #B0B2BC;
    padding: 0.75rem 1rem;
    font-size: 0.87rem;
    color: #444;
    border-radius: 0 4px 4px 0;
    margin: 0.6rem 0;
    line-height: 1.55;
}
/* ── Source cards ────────────────────────────────── */
.ri-source-card {
    background: #ECEEF4;
    border-radius: 6px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.6rem;
    border: 1px solid #C8CAD4;
    border-left: 3px solid #111111;
}
.ri-source-name {
    font-weight: 700;
    color: #111111;
    font-size: 0.93rem;
}
.ri-source-producer {
    font-size: 0.78rem;
    color: #999;
    margin: 0.1rem 0 0.3rem 0;
}
.ri-source-desc {
    font-size: 0.84rem;
    color: #555;
    line-height: 1.45;
}
.ri-source-links {
    margin-top: 0.55rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
    align-items: center;
}
.ri-year-btn {
    display: inline-block;
    padding: 0.18rem 0.52rem;
    background: #0055A4;
    color: #FFFFFF !important;
    border-radius: 4px;
    font-size: 0.74rem;
    font-weight: 600;
    text-decoration: none !important;
    letter-spacing: 0.2px;
    line-height: 1.5;
}
.ri-year-btn:hover { background: #003d7a; }
.ri-src-btn {
    display: inline-block;
    padding: 0.18rem 0.65rem;
    background: #111111;
    color: #FFFFFF !important;
    border-radius: 4px;
    font-size: 0.78rem;
    font-weight: 600;
    text-decoration: none !important;
    line-height: 1.5;
}
.ri-src-btn:hover { background: #333333; }
/* ── Department profile panel ────────────────────── */
.ri-profile-card {
    background: #ECEEF4;
    border-radius: 6px;
    padding: 1rem 1.1rem;
    border: 1px solid #C8CAD4;
    border-top: 3px solid #111111;
}
.ri-profile-title {
    font-weight: 700;
    font-size: 0.95rem;
    color: #111111;
    margin-bottom: 0.4rem;
}
.ri-profile-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.83rem;
    padding: 0.25rem 0;
    border-bottom: 1px solid #F0F0F0;
    color: #555;
}
.ri-profile-row:last-child { border-bottom: none; }
.ri-profile-val {
    font-weight: 600;
    color: #111111;
}
/* ── Validation table ────────────────────────────── */
.ri-val-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}
.ri-val-table th {
    background: #E0E2EA;
    padding: 0.5rem 0.75rem;
    text-align: left;
    font-weight: 700;
    color: #111111;
    border-bottom: 2px solid #C8CAD4;
}
.ri-val-table td {
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid #EBEBF0;
    color: #333;
}
.ri-val-table tr.headline td {
    font-weight: 700;
    color: #0055A4;
    background: #D8E0F4;
}
/* ── Sidebar branding ────────────────────────────── */
.ri-sidebar-brand {
    padding: 0.6rem 0 0.9rem 0;
}
.ri-sidebar-title {
    font-size: 1.65rem;
    font-weight: 900;
    color: #111111;
    letter-spacing: -0.5px;
    line-height: 1.15;
    margin: 0;
}
.ri-sidebar-sub {
    font-size: 0.8rem;
    color: #888;
    margin-top: 0.2rem;
    font-weight: 400;
}
.ri-sidebar-rule {
    height: 3px;
    background: linear-gradient(to right, #0055A4 33.3%, #FFFFFF 33.3% 66.6%, #EF4135 66.6%);
    border: none;
    margin: 0.55rem 0 0 0;
}
/* ── Placeholder box ─────────────────────────────── */
.ri-placeholder {
    background: #FFF8F0;
    border: 2px dashed #EF4135;
    border-radius: 8px;
    padding: 2.5rem;
    text-align: center;
    color: #888;
    font-size: 0.9rem;
}
/* ── Preprint block ──────────────────────────────── */
.ri-preprint {
    background: #ECEEF4;
    border-radius: 6px;
    padding: 1.2rem 1.4rem;
    border: 1px solid #C8CAD4;
    margin-top: 1.5rem;
}
/* ── Info tooltip ────────────────────────────────── */
.ri-info-tip {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #0055A4;
    color: #FFFFFF;
    font-size: 0.6rem;
    font-weight: 700;
    cursor: help;
    position: relative;
    vertical-align: middle;
    margin-left: 3px;
    line-height: 1;
    text-transform: none;
}
.ri-info-tip::after {
    content: attr(data-tip);
    position: absolute;
    bottom: calc(100% + 6px);
    left: 50%;
    transform: translateX(-50%);
    background: #1A1A2E;
    color: #FFFFFF;
    padding: 0.5rem 0.75rem;
    border-radius: 4px;
    font-size: 0.78rem;
    width: 270px;
    text-align: left;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.15s;
    z-index: 9999;
    line-height: 1.45;
    font-weight: 400;
    white-space: pre-line;
}
.ri-info-tip:hover::after {
    opacity: 1;
}
/* ── Limitation list ─────────────────────────────── */
.ri-lim-item {
    padding: 0.55rem 0;
    border-bottom: 1px solid #F0F0F0;
    font-size: 0.88rem;
    color: #444;
    line-height: 1.5;
}
.ri-lim-item:last-child { border-bottom: none; }
.ri-lim-label {
    font-weight: 700;
    color: #111111;
}
/* ── Footer ──────────────────────────────────────── */
[data-testid="stAppViewBlockContainer"],
[data-testid="stMainBlockContainer"],
.block-container {
    padding-bottom: 0 !important;
}
.ri-footer {
    margin-top: 4rem;
    padding: 1.5rem 0 1.5rem 0;
    border-top: 1px solid #C8CAD4;
    text-align: center;
    color: #888;
    font-size: 0.82rem;
}
.ri-footer-copy {
    margin-bottom: 0.4rem;
}
.ri-footer-links {
    display: flex;
    justify-content: center;
    gap: 1rem;
    align-items: center;
}
.ri-footer-links a {
    color: #888 !important;
    text-decoration: none !important;
    font-weight: 500;
}
.ri-footer-links a:hover {
    color: #111111 !important;
}
.ri-footer-sep {
    color: #C8CAD4;
}
</style>
"""


def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


def hero_header(title: str, subtitle: str = ""):
    inject_css()
    st.markdown(f'<h1 class="ri-hero-title">{title}</h1>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="ri-hero-subtitle">{subtitle}</p>', unsafe_allow_html=True)
    st.markdown('<hr class="ri-title-rule">', unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    inject_css()
    st.markdown(f'<h1 class="ri-hero-title">{title}</h1>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="ri-hero-subtitle">{subtitle}</p>', unsafe_allow_html=True)
    st.markdown('<hr class="ri-title-rule">', unsafe_allow_html=True)


def stats_strip(items: list) -> str:
    """items: list of (number, label) or (number, label, red_bool)"""
    cards = ""
    for item in items:
        num, label = item[0], item[1]
        red = item[2] if len(item) > 2 else False
        red_cls = "red" if red else ""
        cards += (
            f'<div class="ri-stat-item">'
            f'<div class="ri-stat-n {red_cls}">{num}</div>'
            f'<div class="ri-stat-l">{label}</div>'
            f'</div>'
        )
    return f'<div class="ri-stats-strip">{cards}</div>'


def source_card(name: str, producer: str, desc: str, links_html: str = "") -> str:
    links_block = f'<div class="ri-source-links">{links_html}</div>' if links_html else ""
    return (
        f'<div class="ri-source-card">'
        f'<div class="ri-source-name">{name}</div>'
        f'<div class="ri-source-producer">{producer}</div>'
        f'<div class="ri-source-desc">{desc}</div>'
        f'{links_block}'
        f"</div>"
    )


def plotly_defaults() -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", color="#1A1A2E"),
        margin=dict(l=10, r=10, t=30, b=10),
    )


def render_footer():
    st.markdown(
        '<div class="ri-footer">'
        '<div class="ri-footer-copy">© 2026 Shail K Patel. Crafted out of boredom.</div>'
        '<div class="ri-footer-links">'
        '<a href="https://github.com/ShailKPatel/Regions-Inegales" target="_blank">GitHub Repo</a>'
        '<span class="ri-footer-sep">|</span>'
        '<span>MIT License</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
