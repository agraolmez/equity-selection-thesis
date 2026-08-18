# Research V6.7 daily MLP overlay result

## Decision

`RETAIN_MONTHLY_REBALANCE_CONTROL_NO_DAILY_MLP_PROMOTION`

The validation-selected neural challenger does not clear every frozen return, Sharpe, drawdown, and paired-inference gate. The monthly portfolio therefore remains primary; the neural daily score may be logged only as a non-trading diagnostic.

## 2023 selection

- Frozen monthly control: -3.82% annualized net excess return, -0.586 Sharpe.
- Frozen V6.6 XGBoost overlay: -2.36%, -0.336 Sharpe.
- Selected V6.7 challenger: `predictive__ensemble_mlp_signed_return_xgb_depth_2__h5__selective` using `ensemble_mlp_signed_return_xgb_depth_2__h5`, -1.71%, -0.232 Sharpe.

## January 2024--August 2025 historical evidence

- Monthly control: 21.63% annualized net excess return, 11.03% volatility, 1.961 Sharpe, -8.21% maximum drawdown.
- Frozen XGBoost overlay: 23.00%, 12.03%, 1.913, -6.58%.
- Selected MLP/ensemble challenger: 25.18%, 11.81%, 2.132, -6.47%.
- Paired annualized difference versus monthly control: 3.55%, HAC t 0.96.
- Paired annualized difference versus frozen XGBoost: 2.18%, HAC t 1.19.

## Interpretation boundary

This study was explicitly opened after the V6.6 historical outcome was known.
It can identify a useful neural mechanism or a paper-trading shadow candidate,
but it cannot create untouched confirmation from January 2024--August 2025.
Prospective evidence can begin no earlier than September 2026.
