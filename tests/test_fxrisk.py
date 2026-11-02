import numpy as np
import pandas as pd
import pytest

from fxrisk import models
from fxrisk.stats import basel_traffic_light, christoffersen_independence, diebold_mariano, kupiec_pof, qlike


@pytest.fixture(scope="module")
def garch_returns() -> pd.Series:
    """Simulated GARCH(1,1) returns with known volatility clustering."""
    rng = np.random.default_rng(0)
    n, omega, alpha, beta = 1500, 0.02, 0.08, 0.9
    r, v = np.zeros(n), np.full(n, omega / (1 - alpha - beta))
    for t in range(1, n):
        v[t] = omega + alpha * r[t - 1] ** 2 + beta * v[t - 1]
        r[t] = np.sqrt(v[t]) * rng.standard_t(8) / np.sqrt(8 / 6)
    return pd.Series(r, index=pd.bdate_range("2015-01-01", periods=n))


def test_qlike_is_zero_at_truth_and_positive_otherwise():
    assert qlike(np.array([2.0]), np.array([2.0]))[0] == pytest.approx(0)
    assert qlike(np.array([2.0]), np.array([1.0]))[0] > 0


def test_diebold_mariano_sign():
    rng = np.random.default_rng(1)
    good, bad = rng.normal(0, 1, 500) ** 2, rng.normal(0, 2, 500) ** 2
    stat, p = diebold_mariano(good, bad)
    assert stat < 0 and p < 0.01


def test_kupiec_accepts_correct_rate_and_rejects_wrong_one():
    rng = np.random.default_rng(2)
    ok = rng.uniform(size=5000) < 0.01
    bad = rng.uniform(size=5000) < 0.03
    assert kupiec_pof(ok, 0.01)[1] > 0.05
    assert kupiec_pof(bad, 0.01)[1] < 0.001


def test_christoffersen_detects_clustering():
    clustered = np.zeros(1000, dtype=int)
    clustered[100:110] = 1
    clustered[500:510] = 1
    assert christoffersen_independence(clustered)[1] < 0.01


def test_basel_zones():
    assert [basel_traffic_light(k) for k in (0, 4, 5, 9, 10)] == ["green", "green", "yellow", "yellow", "red"]


def test_ewma_uses_only_past(garch_returns):
    v = models.ewma_variance(garch_returns)
    shocked = garch_returns.copy()
    shocked.iloc[1000] = 50.0
    v2 = models.ewma_variance(shocked)
    assert v.iloc[1000] == v2.iloc[1000]  # the shock on day t cannot change the forecast for day t
    assert v2.iloc[1001] > v.iloc[1001]


def test_garch_forecasts_are_out_of_sample_and_valid(garch_returns):
    start = str(garch_returns.index[1200].date())
    res = models.garch_forecasts(garch_returns, start, refit=100)
    fc = res.variance.loc[start:]
    assert fc.notna().all() and (fc > 0).all()
    assert res.variance.loc[:start].iloc[:-1].isna().all()
    shocked = garch_returns.copy()
    shocked.iloc[1300] = 50.0
    res2 = models.garch_forecasts(shocked, start, refit=100)
    # refit happened at 1200 and 1300: the forecast for day 1300 must not see the shock
    assert res.variance.iloc[1300] == pytest.approx(res2.variance.iloc[1300])


def test_filtered_hs_var_has_correct_coverage_on_garch_data(garch_returns):
    start = str(garch_returns.index[800].date())
    ewma = models.ewma_variance(garch_returns)
    gjr_t = models.garch_forecasts(garch_returns, start, o=1, refit=100)
    var = models.var_forecasts(garch_returns, start, 0.95, ewma, gjr_t, gjr_t)
    breaches = garch_returns.loc[var.index] > var["Filtered HS (GJR-GARCH)"]
    assert 0.025 < breaches.mean() < 0.08


def test_point_forecasts_shape(garch_returns):
    start = str(garch_returns.index[1300].date())
    fc = models.point_forecasts(garch_returns, start, refit=100)
    assert list(fc.columns) == ["random_walk", "historical_mean", "ar1", "lightgbm"]
    assert fc.notna().all().all()
    assert fc.index[0] == pd.Timestamp(start)
