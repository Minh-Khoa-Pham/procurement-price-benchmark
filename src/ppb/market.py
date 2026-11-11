"""Load public market indices (Brent crude, Henry Hub gas, US CPI) and rebase them."""
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


def _monthly(path: Path, date_col: str, value_col: str, name: str) -> pd.Series:
    df = pd.read_csv(path)
    idx = pd.to_datetime(df[date_col]).dt.to_period("M")
    return pd.Series(df[value_col].values, index=idx, name=name).groupby(level=0).mean()


def load_market(start: str = "2022-01", end: str = "2026-10") -> pd.DataFrame:
    """Monthly Brent (USD/bbl), Henry Hub (USD/MMBtu) and US CPI, aligned on one index."""
    df = pd.concat(
        [
            _monthly(RAW / "brent_monthly.csv", "Date", "Price", "brent"),
            _monthly(RAW / "henry_hub_monthly.csv", "Month", "Price", "gas"),
            _monthly(RAW / "us_cpi.csv", "Date", "Index", "cpi"),
        ],
        axis=1,
    )
    df = df.loc[start:end].ffill()  # CPI is published with a lag: carry the last value
    return df


def rebase(df: pd.DataFrame, base: str = "2022-01") -> pd.DataFrame:
    """Express every column as an index with the base month = 100."""
    return df / df.loc[base] * 100
