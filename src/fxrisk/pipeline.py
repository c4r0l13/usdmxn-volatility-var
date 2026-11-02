"""USD/MXN: (A) can we forecast the rate? (B) volatility forecasting, (C) VaR backtesting.

Run with ``python -m fxrisk.pipeline --data data/raw/daily.csv``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import models, style
from .data import load_rates, log_returns
from .stats import basel_traffic_light, christoffersen_independence, diebold_mariano, kupiec_pof, qlike

TEST_START = "2010-01-01"
EVENTS = {
    "2016-11-09": "US election",
    "2020-03-23": "COVID-19",
    "2024-06-03": "Mexican election",
    "2025-04-07": "Tariff shock",
}


def part_a(r: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    fc = models.point_forecasts(r, TEST_START)
    y = r.loc[fc.index]
    rows = []
    for m in fc.columns:
        err = (y - fc[m]).to_numpy()
        row = {
            "model": m,
            "rmse": float(np.sqrt(np.mean(err**2))),
            "rmse_vs_random_walk": float(np.sqrt(np.mean(err**2)) / np.sqrt(np.mean(y.to_numpy() ** 2))),
            "directional_accuracy": float(np.mean(np.sign(fc[m]) == np.sign(y))) if m != "random_walk" else np.nan,
        }
        if m != "random_walk":
            row["dm_stat"], row["dm_pvalue"] = diebold_mariano(err**2, y.to_numpy() ** 2)
        rows.append(row)
    cum = pd.DataFrame(
        {m: np.cumsum(y.to_numpy() ** 2 - (y - fc[m]).to_numpy() ** 2) for m in fc.columns[1:]}, index=fc.index
    )
    return pd.DataFrame(rows).set_index("model"), cum


def part_b(r: pd.Series) -> tuple[pd.DataFrame, dict[str, pd.Series], models.GarchResult]:
    ewma = models.ewma_variance(r)
    garch = models.garch_forecasts(r, TEST_START, o=0)
    gjr = models.garch_forecasts(r, TEST_START, o=1)
    gjr_skew = models.garch_forecasts(r, TEST_START, o=1, dist="skewt")
    har = models.har_forecasts(r, TEST_START)
    vols = {
        "EWMA (RiskMetrics)": ewma,
        "GARCH(1,1)-t": garch.variance,
        "GJR-GARCH(1,1)-t": gjr.variance,
        "GJR-GARCH(1,1)-skewed-t": gjr_skew.variance,
        "HAR": har,
    }
    idx = r.loc[TEST_START:].index
    sq = (r.loc[idx] ** 2).to_numpy()
    base = qlike(sq, ewma.loc[idx].to_numpy())
    rows = []
    for name, v in vols.items():
        loss = qlike(sq, v.loc[idx].to_numpy())
        row = {
            "model": name,
            "qlike": float(loss.mean()),
            "mse_sq_returns": float(np.mean((sq - v.loc[idx].to_numpy()) ** 2)),
        }
        if name != "EWMA (RiskMetrics)":
            row["dm_stat_vs_ewma"], row["dm_pvalue_vs_ewma"] = diebold_mariano(loss, base)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model"), vols, (gjr, gjr_skew)


def part_c(r: pd.Series, vols: dict, gjrs: tuple) -> tuple[pd.DataFrame, dict[float, pd.DataFrame]]:
    rows, var_tables = [], {}
    for level in (0.95, 0.99):
        var = models.var_forecasts(r, TEST_START, level, vols["EWMA (RiskMetrics)"], *gjrs)
        var_tables[level] = var
        y = r.loc[var.index]
        for m in var.columns:
            breaches = (y > var[m]).to_numpy()
            lr_uc, p_uc = kupiec_pof(breaches, 1 - level)
            lr_ind, p_ind = christoffersen_independence(breaches)
            row = {
                "level": level,
                "method": m,
                "expected_rate": 1 - level,
                "breach_rate": float(breaches.mean()),
                "breaches": int(breaches.sum()),
                "days": len(breaches),
                "kupiec_p": p_uc,
                "independence_p": p_ind,
                "avg_var_pct": float(var[m].mean()),
                "passes_both_5pct": bool(p_uc > 0.05 and p_ind > 0.05),
            }
            if level == 0.99:
                # worst rolling 250-day window, as a regulator would see it
                roll = pd.Series(breaches).rolling(250).sum()
                row["worst_250d_breaches"] = int(roll.max())
                row["worst_250d_zone"] = basel_traffic_light(int(roll.max()))
            rows.append(row)
    return pd.DataFrame(rows), var_tables


# ---------------------------------------------------------------------------- figures


def fig_overview(rates: pd.Series, r: pd.Series, figs: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    axes[0].plot(rates.index, rates, color=style.SERIES[0], lw=1)
    axes[0].set_ylabel("MXN per USD")
    axes[0].set_title("USD/MXN since 1996: calm stretches broken by sharp peso sell-offs")
    axes[1].plot(r.index, r, color=style.SERIES[0], lw=0.5)
    axes[1].set_ylabel("Daily log return (%)")
    axes[1].axvline(pd.Timestamp(TEST_START), color=style.INK_2, lw=1, ls="--")
    axes[1].text(
        pd.Timestamp(TEST_START),
        r.max() * 0.95,
        "  out-of-sample evaluation →",
        fontsize=8,
        color=style.INK_2,
        va="top",
    )
    fig.savefig(figs / "usdmxn_overview.png")
    plt.close(fig)


def fig_point(cum: pd.DataFrame, figs: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.8))
    names = {"historical_mean": "Historical mean", "ar1": "AR(1)", "lightgbm": "LightGBM (lags, momentum, vol)"}
    for (c, label), color in zip(names.items(), style.SERIES[1:4], strict=True):
        ax.plot(cum.index, cum[c], color=color, label=label, lw=1.3)
    ax.axhline(0, color=style.INK_2, lw=1)
    ax.text(cum.index[5], 0, " random walk", va="bottom", fontsize=8, color=style.INK_2)
    ax.set_ylabel("Cumulative squared-error\nadvantage over random walk")
    ax.set_title("No model beats 'tomorrow = today' for the exchange rate itself")
    ax.legend(loc="lower left")
    fig.savefig(figs / "point_forecast_vs_random_walk.png")
    plt.close(fig)


def fig_vol(r: pd.Series, vols: dict, figs: Path) -> None:
    window = slice("2019-10-01", "2020-12-31")
    fig, ax = plt.subplots(figsize=(9, 4))
    ann = np.sqrt(252)
    ax.bar(r.loc[window].index, r.loc[window].abs() * ann, width=1.0, color=style.GRID, label="|daily return| × √252")
    for name, color in (("EWMA (RiskMetrics)", style.SERIES[1]), ("GJR-GARCH(1,1)-t", style.SERIES[0])):
        ax.plot(vols[name].loc[window].index, np.sqrt(vols[name].loc[window]) * ann, color=color, label=name, lw=1.6)
    ax.set_ylabel("Annualized volatility (%)")
    ax.set_title("Volatility forecasts through the COVID-19 shock")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="upper left")
    fig.savefig(figs / "volatility_covid.png")
    plt.close(fig)


def fig_var(r: pd.Series, var99: pd.DataFrame, figs: Path, start: str = "2018-01-01") -> None:
    m_good, m_bad = "Filtered HS (GJR-GARCH)", "Historical simulation (250d)"
    v = var99.loc[start:]
    y = r.loc[v.index]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(y.index, y, color=style.MUTED, lw=0.5, label="Daily return (peso depreciation > 0)")
    ax.plot(v.index, v[m_bad], color=style.SERIES[1], lw=1.2, label=f"99% VaR, {m_bad}")
    ax.plot(v.index, v[m_good], color=style.SERIES[0], lw=1.2, label=f"99% VaR, {m_good}")
    br = y > v[m_good]
    ax.scatter(y.index[br], y[br], s=16, color=style.CRITICAL, zorder=5, label=f"Breach of FHS VaR ({br.sum()})")
    top = max(y.max(), v[m_good].max()) * 1.15
    ax.set_ylim(-4, top)
    for d, label in EVENTS.items():
        x = pd.Timestamp(d)
        if x >= pd.Timestamp(start):
            ax.annotate(
                label,
                (x, top * 0.97),
                xytext=(-3, 0),
                textcoords="offset points",
                fontsize=7.5,
                color=style.INK_2,
                ha="right",
                va="top",
            )
    ax.set_ylabel("%")
    ax.set_title("99% one-day VaR for a short-peso position: HS reacts late, FHS adapts within days")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8)
    fig.savefig(figs / "var_backtest.png")
    plt.close(fig)


def fig_breaches(backtest: pd.DataFrame, figs: Path) -> None:
    b = backtest[backtest.level == 0.99].set_index("method")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    colors = [style.SERIES[0] if ok else style.SERIES[1] for ok in b["passes_both_5pct"]]
    ax.barh(b.index, b["breach_rate"] * 100, color=colors, height=0.6)
    ax.axvline(1.0, color=style.INK_2, ls="--", lw=1)
    ax.text(1.0, -0.65, " target 1%", fontsize=8, color=style.INK_2)
    for yi, (v, ok) in enumerate(zip(b["breach_rate"] * 100, b["passes_both_5pct"], strict=True)):
        ax.text(
            v, yi, f" {v:.2f}%  {'passes' if ok else 'fails'} backtests", va="center", fontsize=8, color=style.INK_2
        )
    ax.set_xlabel("Breach rate of the 99% VaR (%)")
    ax.set_xlim(0, max(b["breach_rate"] * 100) * 1.6)
    ax.set_title("Only filtered historical simulation passes both backtests")
    ax.grid(axis="y", visible=False)
    fig.savefig(figs / "var_breach_rates.png")
    plt.close(fig)


# ---------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/raw/daily.csv")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    out, figs = Path(args.out), Path(args.out) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    style.apply_style()

    rates = load_rates(args.data)
    r = log_returns(rates)
    print(f"{len(r)} returns {r.index[0].date()} -> {r.index[-1].date()}")
    print("A: point forecasts ...")
    a, cum = part_a(r)
    print("B: volatility ...")
    b, vols, gjrs = part_b(r)
    print("C: VaR ...")
    c, var_tables = part_c(r, vols, gjrs)

    fig_overview(rates, r, figs)
    fig_point(cum, figs)
    fig_vol(r, vols, figs)
    fig_var(r, var_tables[0.99], figs)
    fig_breaches(c, figs)

    latest = {name: float(np.sqrt(v.iloc[-1] * 252)) for name, v in vols.items()}
    summary = {
        "sample": {
            "start": str(r.index[0].date()),
            "end": str(r.index[-1].date()),
            "n_returns": len(r),
            "test_start": TEST_START,
            "test_days": int((r.index >= TEST_START).sum()),
        },
        "part_a_point_forecasts": a.round(5).reset_index().to_dict(orient="records"),
        "part_b_volatility": b.round(5).reset_index().to_dict(orient="records"),
        "part_c_var_backtest": c.round(5).to_dict(orient="records"),
        "latest_annualized_vol_forecast_pct": latest,
        "gjr_skewt_last_params": gjrs[1].params,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    cols_c = [
        "level",
        "method",
        "breach_rate",
        "breaches",
        "kupiec_p",
        "independence_p",
        "avg_var_pct",
        "worst_250d_breaches",
        "worst_250d_zone",
    ]
    md = [
        "# Results",
        "",
        f"Out-of-sample: {TEST_START} to {r.index[-1].date()} ({summary['sample']['test_days']} trading days)",
        "",
        "## A. Point forecasts of daily returns",
        "",
        a.round(4).to_markdown(),
        "",
        "## B. One-day volatility forecasts",
        "",
        b.round(4).to_markdown(),
        "",
        "## C. VaR backtests",
        "",
        c[cols_c].round(4).to_markdown(index=False),
        "",
    ]
    (out / "results.md").write_text("\n".join(md) + "\n")
    print(a.round(4).to_string())
    print(b.round(4).to_string())
    print(c[cols_c].round(4).to_string())
    print(latest)


if __name__ == "__main__":
    main()
