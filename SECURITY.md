# Security policy

## Credentials and licensed data

Never commit Alpaca, LSEG, broker, database, or other vendor credentials. Use
the environment-variable names shown in `.env.example`, or the supported local
macOS Keychain integration. Raw CRSP, Compustat, OptionMetrics, LSEG, Alpaca,
and other licensed datasets must remain outside the repository.

If a credential or licensed payload is committed accidentally, revoke or
quarantine it first and then remove it from the complete Git history. Deleting
it only from the latest commit is insufficient.

## Trading boundary

The V6.8 broker integration is research software. It defaults to dry-run and
fails closed unless the frozen current-signal, identity, market, risk, broker,
and credential preflights all pass. Live-money trading is not authorized by
this repository.

## Reporting a vulnerability

Use GitHub's private vulnerability-reporting feature when it is enabled for the
repository. Do not include credentials, licensed data, account identifiers, or
private portfolio positions in a public issue.
