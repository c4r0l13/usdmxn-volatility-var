"""Download daily exchange rates (Federal Reserve H.10 via datahub.io, updated regularly)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/datasets/exchange-rates/main/data/daily.csv"


def main() -> None:
    out = Path("data/raw/daily.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, out)  # always refresh: the series updates daily
    print(f"Data ready: {out.resolve()}")


if __name__ == "__main__":
    main()
