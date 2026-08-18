# Research V6.1 — selector simplification and ensemble study

V6.1 is a separate post-diagnostic successor to the closed V6 selector study.
It asks whether V6's weak tree results can be improved through a broader but
bounded XGBoost search, whether manually removing direct vendor-model outputs
or redundant inputs improves generalization, and whether causal combinations
of XGBoost, MLP, Elastic Net, and direct CAM rank add value.

The completed Step 01 result remains selector-only and immutable.  Following
that result, the user separately authorized Step 02 portfolio research on 13
August 2026.  Step 02 does not retrain or reselect any selector.  Its contract
is `configs/research_v6_1/portfolio_step_02.json`.

## Information views

Every XGBoost, MLP, and Elastic Net comparison is run on the same three views:

1. `full`: all 318 registered raw inputs and their established encoded columns.
2. `cam_free`: all independently constructed inputs except the five direct
   LSEG StarMine Combined Alpha Model level/age/change outputs.
3. `correlation_reduced`: a frozen outcome-blind subset selected only from the
   2015--2019 training inputs. A source is redundant when its rank-transformed
   value has absolute correlation above 0.90 with an already retained source.
   Coverage, registry order, and feature name provide deterministic priority;
   categorical inputs remain eligible and all encoded columns belonging to a
   retained source move together.

No outcome, feature importance, V6 test result, or target identity participates
in correlation reduction. Missing-value handling remains exactly the V6
policy: neutral encoded value plus explicit missingness, with no complete-case
filter.

## Models and objectives

- XGBoost tests pairwise ranking and continuous centered-rank regression. Its
  bounded search extends depth through 6 and varies learning rate, child
  weight, row/column sampling, and L1/L2 regularization. Internal trailing
  holdouts provide early stopping and select the effective tree count. The
  stable XGBoost 3.0 worker represents each month's continuous ordering with
  32 monotonic relevance grades (0--31) for its NDCG ranking implementation;
  this preserves the cross-sectional order at roughly three-percentile
  resolution and avoids treating ranks as exact returns.
- MLP tests RankNet pairwise loss and continuous centered-rank loss, with a
  bounded architecture, dropout, learning-rate, and weight-decay search.
- Elastic Net is the linear continuous-rank benchmark and tunes regularization
  separately on every information view.
- Direct CAM is the point-in-time LSEG CAM country rank itself. Missing CAM is
  neutral, not an exclusion rule. The five-field CAM-only Elastic Net from V6
  remains a different, frozen diagnostic.

For speed and a cleaner comparison, V6.1 preserves the target-specific
refit/history/retune schedules selected using V6's validation window. Within
each feature view, pairwise versus continuous objective and all V6.1
hyperparameters are selected causally from the same label-available internal
holdouts. Every view is carried into the historical test; test outcomes cannot
choose a view or objective.

## Ensembles and controls

The study reports individual models, equal-weight ensemble controls, and
strictly causal dynamic ensembles. Dynamic weights use only prior realized
monthly IC, exponentially decay with a 12-month half-life, require 24 prior
months before adapting, floor negative estimated quality at zero, and remain
bounded. Prespecified combinations include XGBoost+MLP, all three learned
models, each CAM-free learned model plus direct CAM, and all CAM-free learned
models plus CAM. Full-plus-CAM is reported only as a double-counting diagnostic.

The completed V6 scores are merged only as clearly named frozen comparators.
V6.1 reports paired monthly IC changes, seed stability, rank turnover, tail
spreads, and a small label-shuffle smoke control. All candidates remain visible;
there is no winner suppression. The larger grid is disclosed as model
development rather than a single-test inference exercise; historical-test
comparisons are interpreted with explicit selection and multiple-testing
caution.

The prepared outcome-blind views contain 318 raw/850 encoded inputs for
`full`, 313/840 for `cam_free`, and 202/618 for `correlation_reduced`. The last
view removed 106 redundant sources above the 0.90 threshold and 10 sources
that were constant or insufficiently observed in the pre-2020 training slice.

## Evidence boundary and run control

Validation remains January 2020--December 2023 and the historical test remains
January 2024--August 2025. Because V6.1 was designed after examining V6's test,
both are post-diagnostic development evidence. September 2025--August 2026
remains quarantined. Clean confirmation can begin no earlier than September
2026 after the complete selector and portfolio policy is frozen.

The contract was explicitly approved and frozen as `FROZEN_READY_TO_RUN` on
13 August 2026. The full fit is authorized under this exact contract identity.
Readiness can be checked without fitting:

```bash
python scripts/run_v6_1_selector_step_01.py --check-only
```

Every full-run refit block is atomic and resume-safe under contract, code,
input, feature-view, target, profile, and seed identities.

The later $100,000 Alpaca deployment translation is documented separately in
[`ALPACA_PAPER_TRANSLATION_RESULT.md`](ALPACA_PAPER_TRANSLATION_RESULT.md). It
preserves the frozen V6.1 stock decisions and is not a selector or optimizer
rerun.

## Step 02 portfolio contract

The portfolio study keeps the four frozen V6.1 winners and compares four
alpha-gating policies: returns alone, returns plus CAPM alpha, returns plus
FF5+momentum alpha, and returns plus both alphas.  The joint-alpha consensus is
the primary policy.  Long and short candidates are the respective top 20% of
the frozen long-net and short-net ranks.  Alpha acts as a causal sign gate;
position sizing is driven principally by the return-side opportunity.

The study reports equal weights, linear tail ranks, Gaussian tail ranks,
squared tail ranks, and cost/risk optimized weights.  The optimizer maximizes
causal expected net return subject to a 10% primary forecast-volatility budget,
with 8% and 12% robustness budgets for the primary joint-alpha policy.  It
charges spread, square-root impact, financing, and security-level borrow once;
the selector's training cost shadow is not charged again.  FF12 remains
diagnostic-only.  Industry concentration uses point-in-time SIC2, never FF12.

January 2024 through August 2025 is historical development evidence.  Outcomes
from September 2025 through August 2026 remain quarantined, and clean
confirmation can begin no earlier than September 2026 after this complete
policy is frozen.

## Step 02 result

The corrected Step 02 lineage is complete at
`artifacts/research_v6_1/portfolio_step_02/48f429eb0d7f-53a6425a3d92-7256b91cc393`.
All 22 preregistered paths and 440/440 portfolio months completed.  The primary
joint-alpha, 10% risk-budget path earned a 10.39% annualized net excess return,
6.22% realized volatility, and 1.67 net Sharpe under primary costs; under the
frozen cost stress it earned 8.77% and a 1.41 Sharpe.  The primary policy passes
its historical-development promotion gates, but this is not untouched or live
evidence.  The complete result, implementation audit, and interpretation are
recorded in `research_v6_1/PORTFOLIO_STEP_02_RESULT.md`.

## Step 03 optional-short and rank-transfer result

The separately approved Step 03 portfolio-only repair is complete at
`artifacts/research_v6_1/portfolio_step_03/e6253d84f60c-53a6425a3d92-4981f8103e7b`.
It freezes the selectors, removes minimum side-gross requirements, distinguishes
alpha shorts from risk-only hedge shorts, introduces entry/holding bands, and
compares direct-rank, causally shrunk-linear, and broad-bucket payoff mappings.

All six paths and 120/120 portfolio months completed. The best path is the
joint-alpha direct-rank portfolio at 7.03% annualized net excess return and a
1.030 Sharpe (0.926 under stressed costs). It does not beat Step 02. No alpha
short clears the stressed cost hurdle, while the hedge-short sleeve detracts
5.85 percentage points of annualized gross return in the best path. Step 03 is
therefore retained as mechanism and robustness evidence; it does not replace
or retune the Step 02 confirmation candidate. See
[`PORTFOLIO_STEP_03_RESULT.md`](PORTFOLIO_STEP_03_RESULT.md).

## Step 04 optional stock shorts and explicit SPY hedge result

Step 04 isolates the user's final short-sleeve hypotheses without retraining
the selector. It compares a no-floor mean-variance optimizer that may buy
stock covariance hedges against a policy that permits only positive expected
net alpha-stock shorts and uses SPY for residual market-beta hedging. Risk
aversion is selected causally on February 2022--December 2023; the historical
test remains January 2024--August 2025.

Neither policy replaces the user-selected Step 02 FF5/momentum portfolio. The
optional stock-short policy earns 6.80% annualized net excess return at 8.29%
volatility and a 0.821 Sharpe. The positive-alpha-short plus SPY policy earns
3.72% at 6.47% volatility and a 0.575 Sharpe. No stock clears the complete
positive-alpha-short and stressed-cost rule. Stock covariance hedges contribute
-6.37% annualized in the first policy; the SPY hedge contributes -6.04% in the
second. The frozen Step 02 FF5/momentum control remains superior at 9.94% net
excess return, 5.77% volatility, and 1.723 Sharpe. See
[`PORTFOLIO_STEP_04_RESULT.md`](PORTFOLIO_STEP_04_RESULT.md).

## Exact no-forced-short ablation

After clarifying the requested intervention, an exact Step 02 ablation was run
without retraining, retuning, changing candidates, changing payoff mapping, or
introducing another hedge instrument or objective. The explicit 30% short
floor is removed. The maximum-net ceiling is mechanically relaxed from 30% to
100%; otherwise the unchanged 50% minimum long allocation would still force at
least 20% short gross.

Short exposure becomes genuinely discretionary, averaging 22.0%, falling
below 30% in 19/20 months and reaching 6.5%. The result nevertheless weakens:
annualized net excess return is 7.00%, realized volatility is 7.74%, and Sharpe
is 0.904, versus 9.94%, 5.77%, and 1.723 for the original control. Short
contribution improves only 0.22 percentage points annually, while long
contribution falls 3.22 points. The unchanged market-beta limit binds in all
months, so some shorts remain as risk hedges rather than because of a side
floor. See
[`SHORT_FLOOR_ABLATION_RESULT.md`](SHORT_FLOOR_ABLATION_RESULT.md).

## Clean primary-model reproduction

The user-selected FF5+momentum-gated Step 02 portfolio was independently
rebuilt using only the authenticated feature matrix and the frozen model
choices. No fitted selector, prior prediction, or selector checkpoint was
reused. All four rebuilt portfolio signals have zero numerical difference from
the originals, all 20 monthly position files are byte-for-byte identical, and
the complete result reproduces at 9.94% annualized net excess return, 5.77%
volatility, and a 1.723 Sharpe.

This eliminates one-off computational randomness under the frozen three-seed
ensemble. Individual-seed diagnostics remain materially dispersed, with
Sharpes from 0.716 to 1.886, and the historical test contains only 20
post-diagnostic months. The result is reproducible but still requires untouched
forward confirmation. See
[`PRIMARY_REPRODUCTION_RESULT.md`](PRIMARY_REPRODUCTION_RESULT.md).

## Final optimizer challenge and research closure

The final bounded portfolio-maker study held the selectors, features, alpha
gate, candidate tails, constraints, and cost model fixed. It compared the
unchanged optimizer with prespecified expected-return shrinkage,
mean-variance, concentration, and cost-buffer objectives on February
2022--December 2023 validation data. The control had the highest validation
Sharpe at 0.497; no alternative was promoted to the January 2024--August 2025
historical test.

The selected control then reproduced the frozen result exactly: 9.94%
annualized net excess return, 5.77% realized volatility, 1.723 Sharpe, 1.449
stressed-cost Sharpe, -1.96% maximum drawdown, and a 2.641 HAC t-statistic for
mean net excess return. A holdings-aware constraint addendum passes all frozen
rules under practical tolerance. No statistically superior optimizer with a
coherent economic explanation was found. The frozen portfolio is retained and
post-outcome optimizer research is closed. See
[`FINAL_OPTIMIZER_CHALLENGE_RESULT.md`](FINAL_OPTIMIZER_CHALLENGE_RESULT.md).

## User-reopened net-return constraint frontier

On 14 August 2026, the user explicitly reopened a new portfolio-only lineage
to maximize annualized net return using the frozen genuine selectors. This is
not a reinterpretation of the final objective challenge: that study varied
shrinkage, mean-variance, concentration-penalty, and cost-buffer objectives,
whereas the new lineage keeps the original expected-net-return objective and
tests a transparent constraint frontier.

All 25 profiles and 575 validation portfolio-months completed over February
2022--December 2023. Every selection tier chose the same profile: increase the
candidate position cap from 1.0% to 1.5% and change nothing else. Only that
winner and the frozen control were then run over January 2024--August 2025.
The winner earns 14.67% annualized net excess return, 6.32% volatility, a 2.320
Sharpe, a 2.054 stressed-cost Sharpe, and a -1.34% maximum drawdown. Its
total-return CAGR is 21.34%, essentially equal to SPY's 21.33% in the same
window at materially lower volatility. The paired annualized improvement over
the control is 4.73 percentage points (HAC t 5.14; block-bootstrap 95% interval
2.98 to 6.52 points).

The mechanism is modest concentration in the strongest frozen ranks, not
leverage or cheaper costs. The policy remains a 20-month post-diagnostic result,
reaches the 10% ADV boundary at $100 million, and requires untouched
confirmation from September 2026. It is now the strongest historical thesis
candidate. See
[`NET_RETURN_FRONTIER_RESULT.md`](NET_RETURN_FRONTIER_RESULT.md).

## Post-diagnostic constraint-factorial curiosity study

The user subsequently authorized one bounded 2×2×2 curiosity study around the
1.5% candidate-position portfolio. It changed only factor, SIC2-industry, and
one-way ADV limits while preserving every frozen selector, feature, rank,
gate, calibration, cost rule, and portfolio objective. Validation winners were
chosen on February 2022--December 2023 before their January 2024--August 2025
historical outcomes were examined.

The best-return variant relaxes absolute factor limits from 10.0% to 12.5% and
one-way ADV from 10.0% to 12.5%. It earns 15.13% annualized net excess return,
versus 14.67% for the control, but its Sharpe falls from 2.320 to 2.250 and its
paired return improvement is not statistically supported (HAC t 0.416; 95%
block-bootstrap interval -2.07 to +2.24 percentage points). The validation
Sharpe winner that also relaxes industry limits falls to a 1.852 test Sharpe.
The study therefore records a small return frontier but does not replace the
1.5% confirmation candidate. See
[`NET_RETURN_FACTORIAL_EXPLORATORY_RESULT.md`](NET_RETURN_FACTORIAL_EXPLORATORY_RESULT.md).

## Thesis high-return model and Alpaca deployment boundary

The thesis high-return model is the validation-selected `constant_2.05`
balanced tier from the frozen maximum-return scaling frontier. It preserves
the 1.5% portfolio's stocks, signs, and relative weights and earns a 39.17%
historical-test CAGR with a 2.259 Sharpe ratio under the research cost
contract. Its peak gross exposure of 2.62 and peak $100 million ADV use of
24.4% make it a research exposure result rather than a directly deployable
Alpaca portfolio.

The separate Alpaca paper-trading version remains the 1.80-times capped
translation documented in
[`ALPACA_PAPER_TRANSLATION_RESULT.md`](ALPACA_PAPER_TRANSLATION_RESULT.md).
It is an operational broker translation, not the high-return result used in
the thesis.
