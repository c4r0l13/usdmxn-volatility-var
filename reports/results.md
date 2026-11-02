# Results

Out-of-sample: 2010-01-01 to 2026-10-23 (4205 trading days)

## A. Point forecasts of daily returns

| model           |   rmse |   rmse_vs_random_walk |   directional_accuracy |   dm_stat |   dm_pvalue |
|:----------------|-------:|----------------------:|-----------------------:|----------:|------------:|
| random_walk     | 0.737  |                1      |               nan      |  nan      |    nan      |
| historical_mean | 0.7371 |                1.0001 |                 0.4771 |    0.1874 |      0.8513 |
| ar1             | 0.7373 |                1.0004 |                 0.474  |    0.9515 |      0.3414 |
| lightgbm        | 0.7436 |                1.009  |                 0.4942 |    2.8444 |      0.0045 |

## B. One-day volatility forecasts

| model                   |   qlike |   mse_sq_returns |   dm_stat_vs_ewma |   dm_pvalue_vs_ewma |
|:------------------------|--------:|-----------------:|------------------:|--------------------:|
| EWMA (RiskMetrics)      |  1.5177 |           2.0796 |          nan      |            nan      |
| GARCH(1,1)-t            |  1.5122 |           2.0296 |           -0.6747 |              0.4999 |
| GJR-GARCH(1,1)-t        |  1.5069 |           2.0599 |           -0.9407 |              0.3469 |
| GJR-GARCH(1,1)-skewed-t |  1.5074 |           2.0542 |           -0.9287 |              0.3531 |
| HAR                     |  1.5531 |           2.1189 |            2.4439 |              0.0146 |

## C. VaR backtests

|   level | method                       |   breach_rate |   breaches |   kupiec_p |   independence_p |   avg_var_pct |   worst_250d_breaches | worst_250d_zone   |
|--------:|:-----------------------------|--------------:|-----------:|-----------:|-----------------:|--------------:|----------------------:|:------------------|
|    0.95 | Historical simulation (250d) |        0.0576 |        242 |     0.0281 |           0      |        1.2191 |                   nan | nan               |
|    0.95 | EWMA, normal                 |        0.0659 |        277 |     0      |           0.0001 |        1.125  |                   nan | nan               |
|    0.95 | GJR-GARCH, Student-t         |        0.0702 |        295 |     0      |           0.3228 |        1.0776 |                   nan | nan               |
|    0.95 | GJR-GARCH, skewed-t          |        0.0621 |        261 |     0.0005 |           0.2218 |        1.1369 |                   nan | nan               |
|    0.95 | Filtered HS (GJR-GARCH)      |        0.0602 |        253 |     0.0033 |           0.1338 |        1.1527 |                   nan | nan               |
|    0.99 | Historical simulation (250d) |        0.0164 |         69 |     0.0001 |           0.0059 |        1.9373 |                    10 | red               |
|    0.99 | EWMA, normal                 |        0.024  |        101 |     0      |           0.342  |        1.5911 |                    11 | red               |
|    0.99 | GJR-GARCH, Student-t         |        0.0178 |         75 |     0      |           0.7557 |        1.6948 |                     8 | yellow            |
|    0.99 | GJR-GARCH, skewed-t          |        0.0138 |         58 |     0.0194 |           0.8273 |        1.8318 |                     7 | yellow            |
|    0.99 | Filtered HS (GJR-GARCH)      |        0.0119 |         50 |     0.2315 |           0.6274 |        1.9201 |                     7 | yellow            |

