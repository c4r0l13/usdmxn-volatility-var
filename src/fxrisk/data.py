"""USD/MXN daily exchange rate (Federal Reserve H.10 noon buying rates, via FRED/datahub)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://raw.githubusercontent.com/datasets/exchange-rates/main/data/daily.csv"
START = "1996-01-01"  # after the 1994-95 Tequila crisis, once the free float had settled


def load_rates(path: str | Path, country: str = "Mexico", start: str = START) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"])
    s = df.loc[df["Country"] == country].set_index("Date")["Exchange rate"].dropna().sort_index()
    s = s[s > 0]
    s.name = "usdmxn"
    return s.loc[start:]


def log_returns(rates: pd.Series) -> pd.Series:
    """Daily log returns in percent. Positive = peso depreciates."""
    r = 100 * np.log(rates).diff().dropna()
    r.name = "ret"
    return r
