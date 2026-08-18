# V6.8 Alpaca paper deployment

The user authorized V6.8 for prospective paper evaluation on 16 August 2026.
This operating decision does not change the historical research result or its
failed promotion inference gate.

The deployed policy is the frozen 1.80x monthly target portfolio, the selected
V6.7 five-day MLP-XGBoost warning layer, and V6.8 survivor reoptimization after
an executed warning. The adapter is paper-only and fail-closed:

- credentials are read from `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`, with
  dedicated macOS Keychain entries as the local fallback, and are never written
  to artifacts;
- the broker domain is immutably restricted to
  `https://paper-api.alpaca.markets`;
- stale historical weights are rejected;
- a complete, identity-bound signal package generated within 96 hours is
  required;
- starting gross is capped at 1.80x, month-start individual weights at 3.0751%
  (the exact 2.05 x 1.5% translation plus numerical tolerance), and net
  exposure at -0.20 to +0.60;
- new or increased shorts require current tradable, marginable, shortable, and
  easy-to-borrow status;
- orders are whole-share market-on-close orders, reductions are planned first,
  client order IDs are deterministic, and any existing open order blocks a new
  rebalance;
- an account already above the 1.95x safety threshold accepts only a
  reduction-only signal;
- dry-run planning is the default. Submission requires both `--mode submit`
  and the explicit `SUBMIT_V6_8_PAPER_ORDERS` acknowledgement.

Current readiness is **code-complete for the monthly paper chain but blocked for
orders** until a fresh month-end package passes every producer receipt.  The
runtime currently lacks rotated Alpaca paper credentials and a truthful SEC
user agent.  The historical 2025 weights are never a permissible substitute.
The 17 August 2026 package is a mid-month technical smoke only; the first
eligible prospective signal is the final August trading-session close for the
September 2026 book.

The selected V6.7 warning/V6.8 survivor-refill layer is a separate daily
operating layer.  Its exact historical model used same-day Ken French factor
returns at the decision close, while the official files are published with a
lag.  Therefore the monthly portfolio may trade once its gates pass, but the
daily layer must remain shadow-only until an explicitly frozen, live-available
factor-input translation exists.  The implementation must not silently replace
those fields with SPY or another proxy.

## Production feature data

The frozen model consumes 318 registered economic features (850 columns after
the common encoder). The production source contract is
`configs/research_v6_8/production_feature_sources.json`. It assigns every
feature exactly once and forbids silent source substitution:

- 169 features have a same-source refresh path through LSEG, Alpaca news, or
  SEC EDGAR;
- 121 features require a provider bridge with explicit overlap parity against
  their CRSP, Compustat, ownership, or OptionMetrics training definitions;
- 26 factor-dependent fields require a frozen causal signal-close policy;
- the two Pastor-Stambaugh fields remain missing by design, matching every V6
  model month.

The first two market gates are complete. The 10-feature primitive gate passes
all 20 overlap months with 400/400 consolidated mappings. Of the 47 broader
stock-level market formulas, 34 pass in at least 19 of 20 months, 13 fail, and
four cross-sectional market features remain pending. The clearest failed block
is share turnover: LSEG issue shares can combine multiple share classes or ADR
units while CRSP uses the traded security line. Those failed fields must remain
missing until a full-universe security-line bridge passes; current access alone
does not authorize a proxy.

Yahoo Finance is permitted only as a non-authoritative reconciliation check.
The model was trained with missing indicators, so authentic missingness is
acceptable; relabeling a provider proxy as the original feature is not.

Refresh the secret-free readiness audit:

```bash
PYTHONPATH=src /opt/anaconda3/envs/finance/bin/python \
  scripts/audit_v6_8_production_data.py
```

The current report, complete feature matrix, and executed notebook are under
`artifacts/research_v6_8/production_data_readiness/20260816/`.

The checkpointed stock-level market audit and its feature-level decision table
are under `artifacts/research_v6_8/lseg_market_derived_parity/20260817/`.

Local preflight:

```bash
/opt/anaconda3/envs/finance/bin/python scripts/run_v6_8_alpaca_paper_deployment.py
```

Dry-run order plan after a current executable signal exists:

```bash
/opt/anaconda3/envs/finance/bin/python scripts/run_v6_8_alpaca_paper_deployment.py \
  --mode plan --signal /absolute/path/to/current_signal.json
```

Paper submission is intentionally a separate explicit operation. Never place
credentials in a command, config, receipt, notebook, or committed file.

After any submitted paper rebalance, reconcile filled orders and whole-share
positions against the exact rounded target quantities:

```bash
/opt/anaconda3/envs/finance/bin/python scripts/run_v6_8_paper_reconciliation.py \
  --deployment-receipt /absolute/path/to/submitted-run/receipt.json \
  --output /absolute/path/to/post-close-reconciliation
```

The reconciliation passes only when every order is complete and every broker
position equals the planned target. Partial fills remain pending; rejections,
missing orders, fractional positions, and unexplained holdings fail closed.
