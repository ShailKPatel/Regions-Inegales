import os
import sys
import streamlit as st
import pandas as pd

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
from panel_config import PANEL_START, PANEL_END

_MASTER = os.path.join(_HERE, "..", "merged", "france_panel_master.csv")
_POP    = os.path.join(_HERE, "..", "sources", "population_insee.csv")

COLUMN_MAP = {
    "Firm creation rate":    "firm_rate",
    "Median income":         "q2_disp",
    "Higher-ed share":       "edu_share_sup",
    "Unemployment rate":     "unemployment_rate",
    "Doctor density":        "doctor_density_per_100k",
    "% Urban":               "pct_urban",
    "Poverty rate":          "poverty_rate_disp",
    "Gini":                  "gini_disp",
    "Birth rate":            "birth_rate",
    "Death rate":            "death_rate",
    "Marriage rate":         "marriage_rate",
}

IDF_CODES = {"75", "77", "78", "91", "92", "93", "94", "95"}


@st.cache_data(show_spinner=False)
def load_panel() -> pd.DataFrame:
    df = pd.read_csv(_MASTER, sep=";", dtype={"dep_code": object})
    pop = pd.read_csv(_POP, sep=";", dtype={"dep_code": object})
    pop["dep_code"] = pop["dep_code"].str.strip('"')
    df = df.merge(pop[["dep_code", "year", "pop_jan1"]], on=["dep_code", "year"], how="left")
    df = df[(df["year"] >= PANEL_START) & (df["year"] <= PANEL_END)].reset_index(drop=True)
    df["firm_rate"] = df["total_firm_creations"] / df["pop_jan1"] * 1000
    return df


def get_dept_names(df: pd.DataFrame) -> dict:
    """Return {dep_code: dep_name} from the panel."""
    return df[["dep_code", "dep_name"]].drop_duplicates().set_index("dep_code")["dep_name"].to_dict()


def get_year_slice(df: pd.DataFrame, year: int) -> pd.DataFrame:
    return df[df["year"] == year].copy()


def get_dept_year(df: pd.DataFrame, dep_code: str, year: int) -> pd.Series | None:
    rows = df[(df["dep_code"] == dep_code) & (df["year"] == year)]
    return rows.iloc[0] if len(rows) > 0 else None


def get_dept_mean(df: pd.DataFrame, dep_code: str) -> pd.Series | None:
    rows = df[df["dep_code"] == dep_code]
    return rows.mean(numeric_only=True) if len(rows) > 0 else None
