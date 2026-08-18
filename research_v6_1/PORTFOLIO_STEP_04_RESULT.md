# Research V6.1 Step 04 — optional stock shorts and explicit SPY hedge

## Decision

Step 04 is complete. Neither prespecified short-sleeve repair replaces the
user-selected V6.1 Step 02 FF5/momentum-gated optimizer. The frozen Step 02
control remains the working primary portfolio.

The complete run is:

`artifacts/research_v6_1/portfolio_step_04/4a782e4089b6-53a6425a3d92-db0321cf5619`

The frozen contract is
`configs/research_v6_1/portfolio_step_04.json`, SHA-256
`4a782e4089b6c1d52e4836aef9af3fbd661ffb93a89b78180932d985331d7ccb`.
It completed 184/184 validation portfolio-months and 40/40 historical-test
portfolio-months. The selector was not retrained.

## Question and designs

Step 04 asks whether the hedge benefit of shorts can be retained without a
fixed, structurally loss-making stock-short sleeve. Both policies have zero
minimum long gross and zero minimum short gross.

1. `optional_stock_short_utility` maximizes expected net return minus a
   mean-variance penalty. It can accept a stock short with negative standalone
   expected return only when its modeled covariance benefit improves total
   portfolio utility.
2. `positive_alpha_shorts_plus_spy_hedge` permits a stock short only when the
   frozen short rank, negative FF5/momentum alpha gate, and stressed round-trip
   cost hurdle imply positive causal expected net edge. A separately reported
   SPY short then targets market beta within +/-0.02.

A future realized-positive short constraint is deliberately absent because it
would use unavailable future information. Only causal expected net edge can be
constrained at the signal date.

Risk aversion was selected separately for each policy using February
2022--December 2023, before the January 2024--August 2025 historical-test
window. The candidates were 1, 3, 6, and 10. Both policies selected 1. The
validation Sharpe was 0.187 for the optional-stock policy and -0.283 for the
SPY policy, already indicating weak transfer.

## Historical-test result

| Portfolio, January 2024--August 2025 | Annualized net excess return | Realized volatility | Net Sharpe | Stress Sharpe | Max drawdown | Annualized primary cost |
|---|---:|---:|---:|---:|---:|---:|
| Frozen Step 02 FF5/momentum control | 9.94% | 5.77% | **1.723** | **1.449** | -1.96% | 0.99% |
| Optional stock-short utility | 6.80% | 8.29% | 0.821 | 0.685 | -4.45% | 0.90% |
| Positive alpha shorts plus SPY hedge | 3.72% | 6.47% | 0.575 | 0.446 | -3.46% | 0.67% |

The optional-stock policy has mean long gross 0.800, mean stock-short gross
0.261, mean net 0.538, and mean market beta 0.090. The SPY policy has mean long
gross 0.657, no stock-short position, mean SPY short gross 0.386, mean net
0.271, and market beta at the frozen 0.020 limit.

## Sleeve attribution

| Portfolio | Annualized long contribution | Alpha-stock-short contribution | Hedge-stock-short contribution | SPY-hedge contribution | Annualized gross excess return |
|---|---:|---:|---:|---:|---:|
| Frozen Step 02 FF5/momentum control | 14.13% | included in stock shorts | included in stock shorts: -3.19% total | 0.00% | 10.94% |
| Optional stock-short utility | 14.07% | 0.00% | **-6.37%** | 0.00% | 7.71% |
| Positive alpha shorts plus SPY hedge | 10.43% | 0.00% | 0.00% | **-6.04%** | 4.39% |

No stock-month cleared the full positive-alpha-short rule. This is not because
shorts were mechanically prohibited: the optional-stock policy had 202--311
eligible hedge-stock candidates per month and chose mean hedge gross of 26.1%.
Rather, the direct stock-alpha rule found no causal short edge large enough to
clear both the FF5 sign gate and stressed round-trip trading and borrow costs.

## Interpretation

The first hypothesis fails historically. Giving the optimizer freedom to buy
stock covariance hedges did not fix the short sleeve: those hedge stocks cost
6.37 percentage points of annualized gross return and realized volatility rose
relative to the frozen control. The selected risk aversion was low, turnover
remained at its 50% one-way limit, and the market-beta constraint often sat at
its 0.10 boundary. The forecast volatility ceiling itself was not binding.

The second hypothesis also fails as a replacement, but it provides a clean
mechanism result. A separate SPY hedge does reduce realized volatility from the
optional-stock policy's 8.29% to 6.47% and holds beta near 0.02. It is too
expensive in foregone return: the SPY hedge contributes -6.04% annualized, so
the Sharpe remains only 0.575. Market neutrality is therefore not free, and
requiring it more tightly does not improve this return signal.

The frozen Step 02 control remains better despite its negative stock-short
contribution because its short book loses much less (-3.19% rather than about
-6%) while delivering materially stronger diversification: realized
volatility is 5.77%, below both Step 04 policies, and long contribution remains
14.13%. The old short floor was not, by itself, the cause of the superior
Sharpe.

All monthly cost and return identities tie. Maximum gross, turnover, beta, and
other portfolio limits are satisfied up to immaterial solver residue (the
largest displayed boundary overshoot is below 1e-8 in portfolio-weight units).
The solver labels the conic solutions `optimal_inaccurate`; this is the same
practical numerical class accepted by the existing lineage, and the observed
residue cannot change any economic conclusion at the stated $100 million NAV.

## Research boundary

These are historical post-diagnostic development results. September
2025--August 2026 remains quarantined. Clean confirmation remains September
2026 or later after a final policy is frozen. Step 04 does not justify further
post-outcome switching among its two policies.
