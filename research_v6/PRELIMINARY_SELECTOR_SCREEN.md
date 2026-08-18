# V6 preliminary selector screen

> **Inactive predecessor-only screen.** This exploratory run predates the
> completed LSEG/news feature audit and exact December 2019 reconstruction. It
> used only the inherited feature block, is not a result for the current
> 318-feature V6 dataset, and must not be used to promote or reject the real V6
> selector. It is retained only as historical engineering evidence.

## Decision

**Do not promote the non-news baseline to a portfolio backtest.** None of the
four targets has a model whose signal is both positive and persistent across
2020–2023 and 2024–August 2025. The only remaining short-path experiment is the
prespecified incremental Alpaca-news block. A refit/retune cadence search is
deferred because searching cadence on this weak baseline would add computation
and selection risk without a compelling signal.

This is an exploratory screen, not thesis-grade evidence. No optimizer or
portfolio was constructed.

## What was run

- Universe: the inherited monthly U.S.-traded-stock eligibility rules.
- Inputs: all 159 usable predecessor features. The four segment variables were
  excluded only because their verified source table has zero observations.
- Encoded matrix: 543 columns after deterministic preprocessing and missingness
  indicators.
- Targets: CAPM alpha rank, FF5-plus-momentum alpha rank, long net-opportunity
  rank, and short net-opportunity rank.
- Models: pairwise XGBoost, pairwise MLP, a causal trailing-performance ensemble,
  and Elastic Net as the linear benchmark.
- Fit: January 2015–November 2019. Validation: 2020–2023. Historical test:
  January 2024–August 2025.
- Capacity shadow: $100 million NAV, with a conventional $100 million long and
  $100 million short selector shadow. Costs include commissions, spread,
  square-root market impact, short borrow, and a 10% ADV ceiling.

December 2019 Alpaca news is complete, but the December 2019 predecessor
predictor cross-section is not yet available. It will be built as a new V6
point-in-time month; it is never interpolated from adjacent months.

## Results

Values below are mean monthly cross-sectional rank IC. A positive value is
better.

| Target | Model | 2020–2023 | 2024–Aug. 2025 | Interpretation |
|---|---:|---:|---:|---|
| CAPM alpha | Elastic Net | -0.0004 | +0.0193 | Test mean is driven by unstable months; median test IC is -0.0255. |
| CAPM alpha | XGBoost | -0.0159 | -0.0119 | Negative in both periods. |
| CAPM alpha | MLP | -0.0022 | -0.0015 | Approximately zero and negative. |
| CAPM alpha | Dynamic ensemble | -0.0139 | -0.0122 | Dynamic weighting does not rescue the components. |
| FF5+momentum alpha | Elastic Net | +0.0196 | -0.0006 | Validation effect does not persist. |
| FF5+momentum alpha | XGBoost | +0.0037 | -0.0193 | Sign reversal. |
| FF5+momentum alpha | MLP | +0.0004 | -0.0007 | Approximately zero. |
| FF5+momentum alpha | Dynamic ensemble | +0.0008 | -0.0151 | No persistence. |
| Long net rank | Elastic Net | +0.0120 | +0.0032 | Small positive mean, but test median IC is negative. |
| Long net rank | XGBoost | -0.0243 | -0.0342 | Negative in both periods. |
| Long net rank | MLP | -0.0479 | +0.0104 | Large regime/model reversal. |
| Long net rank | Dynamic ensemble | -0.0486 | -0.0196 | Negative in both periods. |
| Short net rank | Elastic Net | -0.0163 | -0.0189 | Negative in both periods. |
| Short net rank | XGBoost | +0.0337 | -0.0006 | The strongest validation hint disappears in test. |
| Short net rank | MLP | -0.0462 | +0.0206 | Sign reversal in the opposite direction. |
| Short net rank | Dynamic ensemble | -0.0067 | +0.0135 | Test median IC remains negative and only 45% of months are positive. |

The first exploratory short-cost output is explicitly invalidated and retained
only for audit history. The table uses the corrected V2 calculation, in which
position size is based on the eligible count for each side separately.

## Missing values and preprocessing

Complete-case trading is impossible: after excluding the four structurally
empty segment variables, **zero stocks in every month have all 159 inputs
observed**. The average inherited long-eligible universe is about 2,180 stocks
per month. Dropping incomplete rows would therefore eliminate the universe and
systematically favor firms covered by optional datasets.

The implemented alternative is:

- continuous firm features: within-month 1st/99th percentile clipping, then a
  centered cross-sectional rank; missing becomes the neutral value zero and a
  separate missing indicator is added;
- binary features: retain 0/1, set missing to zero, and add a missing indicator;
- categorical and multi-category features: use the frozen predecessor
  vocabulary with an explicit unknown state;
- aggregate market-state features: standardize using strictly earlier signal
  dates only, clip extreme z-scores to ±5, and add missing indicators.

This lets XGBoost and the MLP learn whether missingness matters without
pretending that a missing raw value equals an economic zero. Elastic Net can
shrink weak encoded predictors to zero; tree and neural models use
regularization rather than literal zero coefficients.

## Brief feature review

The retained registry is broad rather than hand-selected: 20 price/trend, 20
risk/return, 13 size/liquidity/friction, 26 option, 23 aggregate-state, 29
valuation/fundamental, and 28 ownership/filing/event/identity features. This is
well aligned with the modern empirical-asset-pricing approach of supplying
many economically motivated characteristics to regularized nonlinear models.
Gu, Kelly, and Xiu find trees and neural networks useful precisely because of
nonlinear interactions, with momentum, liquidity, and volatility among the
dominant signals ([RFS, 2020](https://academic.oup.com/rfs/article/33/5/2223/5758276)).
The valuation, profitability, investment, size, and factor-exposure fields map
to the economic dimensions in the Fama–French five-factor model
([JFE, 2015](https://www.sciencedirect.com/science/article/pii/S0304405X14002323)).

The news addition is also academically motivated, but its incremental value
must be tested rather than assumed. Tetlock documents return and volume effects
from quantified media tone ([Journal of Finance, 2007](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2007.01232.x)).
V6 will use an article only for directly tagged stocks, authenticated corporate
family, or a separately identified related industry; it will not broadcast all
news to all stocks. Creation and usable update timestamps must both be no later
than signal close.

No new economic characteristic should be added merely because the baseline is
weak. The next decision is narrow: does the frozen, point-in-time news block
create a stable incremental selector signal? If not, V6 stops before portfolio
optimization.
