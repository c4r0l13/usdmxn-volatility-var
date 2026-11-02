"""Forecast-comparison and VaR-backtesting statistics."""

from __future__ import annotations

import numpy as np
from scipy import stats


def qlike(realized_sq: np.ndarray, var_forecast: np.ndarray) -> np.ndarray:
    """QLIKE loss (Patton, 2011): robust to noise in the volatility proxy."""
    ratio = np.asarray(realized_sq) / np.asarray(var_forecast)
    return ratio - np.log(np.clip(ratio, 1e-12, None)) - 1


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> tuple[float, float]:
    """DM test with the Harvey-Leybourne-Newbold small-sample correction.

    Returns (statistic, two-sided p-value). Negative statistic = model A has lower loss.
    """
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)
    d_mean = d.mean()
    gamma = [np.sum((d[k:] - d_mean) * (d[: n - k] - d_mean)) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    dm = d_mean / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    return float(stat), float(2 * stats.t.sf(abs(stat), df=n - 1))


def kupiec_pof(breaches: np.ndarray, alpha: float) -> tuple[float, float]:
    """Kupiec (1995) proportion-of-failures test. H0: breach rate == alpha."""
    x = int(np.sum(breaches))
    n = len(breaches)
    p_hat = x / n
    if x == 0:
        lr = -2 * n * np.log(1 - alpha)
    elif x == n:
        lr = -2 * n * np.log(alpha)
    else:
        lr = -2 * ((n - x) * np.log(1 - alpha) + x * np.log(alpha) - (n - x) * np.log(1 - p_hat) - x * np.log(p_hat))
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen_independence(breaches: np.ndarray) -> tuple[float, float]:
    """Christoffersen (1998) test: are breaches clustered? H0: independent."""
    b = np.asarray(breaches).astype(int)
    prev, curr = b[:-1], b[1:]
    n00 = np.sum((prev == 0) & (curr == 0))
    n01 = np.sum((prev == 0) & (curr == 1))
    n10 = np.sum((prev == 1) & (curr == 0))
    n11 = np.sum((prev == 1) & (curr == 1))
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def ll(p, a, b_):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return a * np.log(1 - p) + b_ * np.log(p)

    lr = -2 * (ll(pi, n00 + n10, n01 + n11) - ll(pi0, n00, n01) - ll(pi1, n10, n11))
    return float(lr), float(stats.chi2.sf(lr, 1))


def basel_traffic_light(breaches_last_250: int) -> str:
    """Basel zones for a 99% VaR over 250 days."""
    if breaches_last_250 <= 4:
        return "green"
    return "yellow" if breaches_last_250 <= 9 else "red"
