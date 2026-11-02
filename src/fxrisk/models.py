"""Rolling-origin (walk-forward) forecasts.

Every model is re-estimated every ``refit`` trading days on an expanding
window and only uses information available at the close of day t to
forecast day t+1. No forecast ever sees its target.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from lightgbm import LGBMRegressor
from scipy import stats

REFIT = 21  # roughly monthly


def _refit_points(index: pd.DatetimeIndex, start: str, refit: int) -> list[int]:
    first = int(index.searchsorted(pd.Timestamp(start)))
    return list(range(first, len(index), refit))


# ---------------------------------------------------------------------------- Part A: returns


def lag_features(r: pd.Series, lags: int = 10) -> pd.DataFrame:
    X = pd.concat({f"lag{k}": r.shift(k) for k in range(1, lags + 1)}, axis=1)
    X["vol5"] = r.rolling(5).std().shift(1)
    X["vol22"] = r.rolling(22).std().shift(1)
    X["mom22"] = r.rolling(22).sum().shift(1)
    return X


def point_forecasts(r: pd.Series, start: str, refit: int = REFIT, seed: int = 0) -> pd.DataFrame:
    """Next-day return forecasts. Row t holds the forecast of r_t made at t-1."""
    X = lag_features(r)
    out = pd.DataFrame(index=r.index, columns=["random_walk", "historical_mean", "ar1", "lightgbm"], dtype=float)
    out["random_walk"] = 0.0
    points = _refit_points(r.index, start, refit)
    for i, p in enumerate(points):
        end = points[i + 1] if i + 1 < len(points) else len(r)
        hist = r.iloc[:p]
        out.iloc[p:end, out.columns.get_loc("historical_mean")] = hist.mean()
        y, x = hist.iloc[1:].to_numpy(), hist.iloc[:-1].to_numpy()
        beta = np.polyfit(x, y, 1)
        prev = r.shift(1).iloc[p:end].to_numpy()
        out.iloc[p:end, out.columns.get_loc("ar1")] = beta[0] * prev + beta[1]
        Xtr = X.iloc[:p].dropna()
        model = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=7,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            random_state=seed,
            verbose=-1,
        )
        model.fit(Xtr, r.loc[Xtr.index])
        out.iloc[p:end, out.columns.get_loc("lightgbm")] = model.predict(X.iloc[p:end])
    return out.iloc[points[0] :]


# ---------------------------------------------------------------------------- Part B: volatility


def ewma_variance(r: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics EWMA. Row t = variance forecast for day t made at t-1."""
    v = np.empty(len(r))
    v[0] = r.iloc[:22].var()
    rv = r.to_numpy()
    for t in range(1, len(r)):
        v[t] = lam * v[t - 1] + (1 - lam) * rv[t - 1] ** 2
    return pd.Series(v, index=r.index, name="ewma")


@dataclass
class GarchResult:
    """Row t holds what was known at the close of t-1."""

    variance: pd.Series
    nu: pd.Series  # Student-t degrees of freedom (inf for a normal model)
    resid_quantiles: pd.DataFrame  # empirical quantiles of past standardized residuals (for FHS)
    dist_quantiles: pd.DataFrame  # quantiles of the fitted innovation distribution
    params: dict  # parameters of the last fit


def garch_forecasts(
    r: pd.Series,
    start: str,
    o: int = 0,
    dist: str = "t",
    refit: int = REFIT,
    levels: tuple[float, ...] = (0.95, 0.99),
) -> GarchResult:
    """(GJR-)GARCH(1,1) one-step variance forecasts, re-estimated monthly."""
    var = pd.Series(np.nan, index=r.index)
    nu = pd.Series(np.nan, index=r.index)
    rq = pd.DataFrame(np.nan, index=r.index, columns=list(levels))
    dq = pd.DataFrame(np.nan, index=r.index, columns=list(levels))
    points = _refit_points(r.index, start, refit)
    model = arch_model(r, mean="Zero", vol="GARCH", p=1, o=o, q=1, dist=dist, rescale=False)
    for i, p in enumerate(points):
        end = points[i + 1] if i + 1 < len(points) else len(r)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit(last_obs=r.index[p], disp="off", show_warning=False)
        # fixed parameters, filtered forward: forecasts for days p ... end-1
        fc = res.forecast(start=r.index[p - 1], align="target", horizon=1, reindex=True).variance["h.1"]
        var.iloc[p:end] = fc.iloc[p:end].to_numpy()
        nu.iloc[p:end] = res.params.get("nu", res.params.get("eta", np.inf))
        z = (r.iloc[:p] / res.conditional_volatility.iloc[:p]).dropna().to_numpy()
        dist_params = res.params[res.model.distribution.parameter_names()].to_numpy() if dist != "normal" else None
        for lv in levels:
            rq.iloc[p:end, rq.columns.get_loc(lv)] = np.quantile(z, lv)
            dq.iloc[p:end, dq.columns.get_loc(lv)] = float(res.model.distribution.ppf(lv, dist_params))
    return GarchResult(var, nu, rq, dq, res.params.round(4).to_dict())


def har_forecasts(r: pd.Series, start: str, refit: int = REFIT) -> pd.Series:
    """HAR model (Corsi, 2009) on squared returns: daily, weekly and monthly components."""
    sq = r**2
    feats = pd.DataFrame({"d": sq, "w": sq.rolling(5).mean(), "m": sq.rolling(22).mean()}).shift(1)
    out = pd.Series(np.nan, index=r.index)
    floor = 0.05 * sq.median()
    points = _refit_points(r.index, start, refit)
    for i, p in enumerate(points):
        end = points[i + 1] if i + 1 < len(points) else len(r)
        tr = feats.iloc[:p].dropna()
        A = np.column_stack([np.ones(len(tr)), tr.to_numpy()])
        coef, *_ = np.linalg.lstsq(A, sq.loc[tr.index].to_numpy(), rcond=None)
        Xf = np.column_stack([np.ones(end - p), feats.iloc[p:end].to_numpy()])
        out.iloc[p:end] = np.clip(Xf @ coef, floor, None)
    return out


# ---------------------------------------------------------------------------- Part C: VaR


def var_forecasts(
    r: pd.Series, start: str, level: float, ewma: pd.Series, gjr_t: GarchResult, gjr_skewt: GarchResult
) -> pd.DataFrame:
    """Upper-tail VaR (loss when the peso depreciates), in % of position, one day ahead."""
    idx = r.loc[start:].index
    sigma = np.sqrt(gjr_skewt.variance.loc[idx])
    out = pd.DataFrame(index=idx)
    out["Historical simulation (250d)"] = r.rolling(250).quantile(level).shift(1).loc[idx]
    out["EWMA, normal"] = np.sqrt(ewma.loc[idx]) * stats.norm.ppf(level)
    out["GJR-GARCH, Student-t"] = np.sqrt(gjr_t.variance.loc[idx]) * gjr_t.dist_quantiles.loc[idx, level]
    out["GJR-GARCH, skewed-t"] = sigma * gjr_skewt.dist_quantiles.loc[idx, level]
    out["Filtered HS (GJR-GARCH)"] = sigma * gjr_skewt.resid_quantiles.loc[idx, level]
    return out
