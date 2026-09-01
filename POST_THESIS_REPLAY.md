# Post-thesis V6.8 operational replay

This addendum documents the 111.63% cumulative-return figure associated with
the V6.8 project. It was produced after the thesis was finalized and does not
modify the thesis, its research conclusions, or its evidence boundary.

## Result

| Metric | January 2024-July 2026 |
|---|---:|
| Holding months | 31 |
| Cumulative net total return | 111.63% |
| $100,000 ending wealth | $211,633.66 |
| Net total-return CAGR | 33.67% |
| Annualized net excess return | 25.50% |
| Annualized excess-return volatility | 12.58% |
| Monthly net-excess Sharpe | 2.028 |
| Maximum monthly drawdown | -3.37% |
| Worst reported segment daily drawdown | -9.24% |

The cumulative return is the compounded product of the 31 monthly `net_return`
observations. CAGR uses 31/12 years. Annualized excess return is 12 times the
mean monthly `net_excess_return`; annualized volatility is its sample standard
deviation times the square root of 12; Sharpe is their ratio.

## Evidence boundary

The series stitches two distinct bodies of evidence:

1. **January 2024-August 2025:** the closed V6.8 historical-development
   evaluation from run
   `d55a1db689f9-2ef39b0755c5-2e422abd519f`. This period was already known when
   later operating decisions were made.
2. **September 2025-July 2026:** a retrospective operational reconstruction of
   the same V6.8 survivor-reoptimization policy. This segment is diagnostic,
   not a record of orders generated contemporaneously in paper or live trading.

The combined 31-month number is useful for describing how the frozen mechanism
behaved across the reconstructed period. It is not clean prospective evidence
and must not be described as realized brokerage performance.

## Required limitations

- The September 2025-July 2026 segment was reconstructed after its outcomes
  existed.
- July 2026 and the 3 August exit use SPY as the daily market-factor proxy,
  zero UMD, and the last available official daily risk-free rate because the
  official July factor file was not yet published.
- Historical borrow eligibility is modelled rather than backed by a
  contemporaneous broker locate archive.
- The -9.24% drawdown is the worse of the two segment-level daily drawdowns. A
  single uninterrupted public daily ledger is not distributed.
- No Alpaca order is represented by this replay.

## Reviewable files

- [`monthly_returns.csv`](evidence/post_thesis_v6_8_replay/monthly_returns.csv)
  contains the complete 31-month series and labels each row by evidence status.
- [`summary.json`](evidence/post_thesis_v6_8_replay/summary.json) contains the
  formulas, source identities, headline metrics, and caveats.
- `tests/unit/test_post_thesis_v6_8_evidence.py` independently recomputes the
  public metrics from the checked-in CSV.

The underlying licensed and generated parquet ledgers remain excluded from
Git. Their SHA-256 identities are recorded in `summary.json`.
