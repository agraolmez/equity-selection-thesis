# V6 feature audit and backtest-input decision

Audit completed on **12 August 2026**.

## Decision

The feature and label inputs are ready for a V6 selector backtest. This is a
data-readiness decision, not evidence that any selector predicts returns and
not authorization to construct a portfolio. No selector backtest was run as
part of this audit.

The primary model input is the complete registered set of **318 raw features**
over **278,897 eligible stock-months** and **128 holding months** from January
2015 through August 2025. December 2019 is present and was reconstructed from
point-in-time sources without interpolation.

## Features that will be used

| Family | Count | Information represented |
|---|---:|---|
| Price trends and path | 22 | Multi-horizon returns, momentum, reversal, moving-average distance, trend slope/R², drawdown, industry momentum and momentum acceleration |
| Risk and return distribution | 20 | Realized/downside volatility, trailing factor betas, idiosyncratic volatility, skew, kurtosis, return extremes |
| Size, liquidity, and frictions | 14 | Price, market cap, dollar volume, turnover level/volatility, Amihud illiquidity, spread, zero trading, volume shock, listing age |
| Valuation and fundamentals | 33 | Valuation ratios, profitability, margins, accruals, investment, leverage, liquidity, issuance/payout, growth, quarterly earnings/sales momentum, report recency, loss and source flags |
| Ownership, filings, events, and identity | 27 | Institutional ownership, SEC filing age/count/content, restatement flags, ADR/domicile/exchange/SIC/reporting-currency categories |
| OptionMetrics | 26 | Put/call volume and open interest, ratios, changes, option-to-stock volume, coverage age, historical-volatility term structure |
| Aggregate state | 23 | Trailing factor returns, risk-free rate, market volatility, Pástor–Stambaugh liquidity state and innovation |
| LSEG analyst expectations | 97 | FY1/FY2/FQ1 consensus levels, dispersion, estimate counts and SmartEst; normalized forward value/growth; recommendation and price-target revisions; CAM rank and causal changes |
| Alpaca related news | 56 | Direct-stock, corporate-family, and SIC2-industry news over 1/7/30 days: count, LM tone, uncertainty, recency, duplication, content availability, and source/relationship flags |
| **Total** | **318** | |

The exact column-by-column authority is
`/Users/agraemirolmez/Desktop/Thesis/processed/research_v6/model_features_v2/feature_registry.csv`.
It records each feature's family, kind, source column, unit, transformation,
and membership in each prespecified information view.

The primary view is `primary_full` (318). The two controlled comparisons are
`cam_free` (313) and `cam_only` (5: CAM information age, country rank, and
causal 1/3/6-month rank changes). The CAM views isolate the value of LSEG's
vendor composite rather than allowing it to blur the independent-feature
result.

## Approved engineered extension

Fifteen outcome-blind features were added on 12 August 2026 without changing
any of the original 303 values:

- price/peer: value-weighted SIC2 industry momentum and six-month momentum
  acceleration;
- liquidity: 63-session standard deviation of daily share turnover;
- analyst expectations: forward FY1 earnings yield, FY2-versus-FY1 expected
  EPS and revenue growth, price-target upside, exact-calendar 1/3-month target
  revisions, and direction-normalized 1/3-month recommendation upgrades;
- quarterly fundamentals: earnings and sales changes versus the same fiscal
  quarter one year earlier divided by signal-date market capitalization,
  consecutive earnings increases up to eight quarters, and days since the
  latest publicly available quarterly report.

The industry momentum follows the conventional value-weighted industry-return
interpretation. Price-target and recommendation changes require the exact same
PERMNO one or three calendar months earlier; gaps remain missing. A positive
recommendation change means an upgrade on LSEG's 1=Strong Buy through 5=Strong
Sell scale. Naive FY1 EPS level changes were not added because a fiscal-horizon
rollover could otherwise be misclassified as a forecast revision.

## Exclusions

The following are deliberately not model inputs:

- `business_segment_count`, `geographic_segment_count`,
  `business_segment_revenue_hhi`, and
  `business_segment_revenue_hhi_change_1y`: the authenticated segment source
  has zero observations, so retaining them would create four synthetic columns.
- `ff12_sector_category`: diagnostic-only under the project governance rule;
  it cannot form, neutralize, or weight portfolios.
- PERMNO, PERMCO, ticker, issuer name, eligibility flags, and other identifiers
  as predictors. The first two remain join keys only.
- Raw LSEG vendor dates. They are converted to nonnegative information-age
  fields.
- Local-currency LSEG monetary levels. The 52 currency-sensitive fields are
  sourced from the separate authenticated `Curn=USD` download.
- Raw article text as a high-dimensional model input. Available content is
  deduplicated and summarized with the finance-specific Loughran–McDonald
  dictionary; `full_content_share` records content availability.

## Missingness and preprocessing

There are **zero complete cases** across all 318 raw features in every month.
A complete-case rule would therefore eliminate the trading universe and select
firms according to optional-vendor coverage. V6 retains the full eligible
universe and uses one common, outcome-blind encoder:

- continuous firm variables: monthly 1%/99% clipping, centered cross-sectional
  rank, neutral zero for missing, plus an explicit missing indicator;
- binary variables: 0/1 value, neutral zero for missing, plus an indicator;
- categorical/multihot variables: frozen vocabulary with an unknown state;
- aggregate variables: strictly past expanding z-score, clipped to [-5, 5],
  plus an indicator.

The primary 318 raw inputs become **850 finite encoded columns**. `cam_free`
becomes 840 and `cam_only` becomes 10. This preprocessing does not inspect any
future outcome. It lets Elastic Net shrink weak encoded inputs to zero and lets
XGBoost/MLP regularization decide which nonlinear inputs matter.

## Quality evidence

| Check | Result |
|---|---:|
| Canonical monthly feature files | 128/128 |
| Rows | 278,897 |
| Duplicate month–PERMNO keys | 0 |
| Infinite numeric values | 0 |
| Receipt/hash failures | 0 |
| Original V1 feature values changed | 0 |
| All-empty included features | 0 |
| Forbidden eligibility/FF12/target columns in feature panel | 0 |
| Months with all four target ranks | 128/128 |
| December 2019 eligible feature rows | 2,051 |
| December 2019 valid target rows | 2,050 |

Average feature availability by family ranges from **87.3%** for valuation and
fundamentals and **88.3%** for LSEG to above **95%** for the main market-data
families. News columns have complete row coverage because neutral no-related-
news values and a separate source-availability flag are explicit. Some
individual optional-source features are sparse; that is why their missingness
is modeled instead of filtering stocks.

Six retained fields have no level variation over this window:
`accounting_market_currency_verified`, `optionmetrics_covered_indicator`, the
FY1/FY2/FQ1 LSEG information ages, and the CAM information age. Their value
columns become neutral under cross-sectional ranking, although their
missingness indicators can still vary. They remain registered because the
approved policy is to supply all valid features and let regularization remove
uninformative inputs. They must not later be presented as economically
important unless a new vintage creates genuine variation.

The engineered-feature mean coverage ranges from **85.2% to 100.0%**. Excluding
the structurally missing initial lag months, the lowest monthly coverage is
78.3% for a three-month analyst-revision feature; expected revenue growth has a
minimum of 81.0%.
Quarterly earnings/sales momentum has approximately 94.5% mean coverage,
industry momentum 99.8%, and turnover volatility 100.0% after rounding.

Thirty-eight stale Finder-style duplicate V1 month files were previously moved,
not deleted, to `model_features_v1/quarantine/stale_duplicate_month_files`.
The V2 production month directory contains exactly the 128 receipt-bound files.

## Point-in-time controls

- Monthly action cutoff: signal close.
- Industry momentum uses only the signal-date cross-section and each stock's
  already-trailing 12-minus-1 momentum.
- Turnover volatility uses exactly the 63 trading sessions ending on the signal
  date and date-valid CRSP shares outstanding.
- Quarterly features admit only observations whose observation and public
  availability dates are not after the signal, resolve fiscal-calendar versions
  as of that signal, and reject stale anchors.
- News: both `created_at` and `updated_at` must be available by that cutoff.
- News propagation: only direct-stock, authenticated corporate-family, or
  related SIC2-industry mappings; no market-wide article is assigned to every
  stock.
- LSEG: historical CUSIP9 addressing and exact authenticated signal date.
- CAM changes: same-PERMNO lags of 1, 3, and 6 months only.
- December 2019: signal 29 November 2019, entry 2 December 2019, exit 2 January
  2020; exact formulas and realized returns, no interpolation.
- Labels are separate from features. Four higher-is-better monthly ranks are
  available: CAPM alpha, FF5+momentum alpha, long net opportunity, and short
  net opportunity.

## Academic and professional alignment

The broad characteristic design is consistent with Gu, Kelly, and Xiu's
machine-learning asset-pricing framework, where nonlinear trees and neural
networks ingest many economically motivated characteristics and momentum,
liquidity, and volatility are especially important
([Review of Financial Studies, 2020](https://academic.oup.com/rfs/article/33/5/2223/5758276)).

News tone uses a finance-specific dictionary rather than a generic sentiment
lexicon, following Loughran and McDonald's warning that ordinary negative-word
lists are poorly suited to financial language
([Journal of Finance, 2011](https://doi.org/10.1111/j.1540-6261.2010.01625.x)).
Textual return predictability is a legitimate challenger, but the current
dictionary summaries are a transparent baseline—not a claim to match a
supervised language model such as Ke, Kelly, and Xiu
([NBER Working Paper 26186](https://www.nber.org/papers/w26186)). Analyst
forecasts, recommendations, and price targets are also retained because the
literature finds analyst reports contain price-relevant information
([NBER Working Paper 9246](https://www.nber.org/papers/w9246)).

The design is therefore broad enough for the intended XGBoost/MLP comparison,
while keeping coverage flags, currency treatment, as-of dates, and vendor-model
content visible—controls expected in a professional research pipeline.

## Authorities

- Model panel and exact registry:
  `/Users/agraemirolmez/Desktop/Thesis/processed/research_v6/model_features_v2`
- Build identity:
  `01c2e95ca6b928446b3524f1c621db01cc6285858ca39ac7646441d1265de2d7`
- December reconstruction:
  `/Users/agraemirolmez/Desktop/Thesis/processed/research_v6/december_2019_v1`
- Point-in-time news block:
  `/Users/agraemirolmez/Desktop/Thesis/processed/research_v6/news_features_v2`
- LSEG USD raw repair:
  `/Users/agraemirolmez/Desktop/Thesis/raw/research_v6/lseg_usd_v1`

The selector contract and resumable production runner are now frozen and the
full 278,897-by-850 encoded input cache has been prepared without fitting a
model. The next action is the real selector backtest under
`configs/research_v6/selector_step_01.json`. Portfolio construction and
optimization remain out of scope.
