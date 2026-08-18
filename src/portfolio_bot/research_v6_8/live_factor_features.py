"""Official, point-in-time factor snapshot and live V6.8 factor features."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import urllib.request
import zipfile
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from portfolio_bot.features.feature_math import ewls_factor_fit


FACTOR_NAMES = ("mktrf", "smb", "hml", "rmw", "cma", "umd")
BETA_NAMES = tuple(f"beta_label_{name}" for name in FACTOR_NAMES)
TRAILING_FACTOR_NAMES = tuple(
    f"factor_{factor}_trailing_{months}m"
    for months in (1, 3, 12)
    for factor in FACTOR_NAMES
)


class LiveFactorFeatureError(RuntimeError):
    """The official factor snapshot or causal feature policy failed."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n").encode(
            "utf-8"
        ),
    )


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _download(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Bocconi-thesis-paper-research/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    if not payload:
        raise LiveFactorFeatureError(f"empty official factor response: {url}")
    return payload


def _zip_text(payload: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise LiveFactorFeatureError("official factor archive must contain one CSV")
        return archive.read(members[0]).decode("utf-8", errors="strict")


def _parse_factor_csv(
    payload: bytes, *, date_digits: int, value_names: tuple[str, ...]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(rf"^\d{{{date_digits}}}$")
    for line in _zip_text(payload).splitlines():
        fields = [item.strip() for item in line.split(",")]
        if not fields or not pattern.fullmatch(fields[0]):
            continue
        if len(fields) < len(value_names) + 1:
            raise LiveFactorFeatureError("official factor row has too few columns")
        try:
            values = [float(fields[index + 1]) / 100.0 for index in range(len(value_names))]
        except ValueError as error:
            raise LiveFactorFeatureError("invalid numeric official factor value") from error
        stamp = pd.to_datetime(
            fields[0], format="%Y%m%d" if date_digits == 8 else "%Y%m"
        )
        rows.append(
            {
                "period_key": fields[0],
                "source_date": stamp,
                **dict(zip(value_names, values, strict=True)),
            }
        )
    if not rows:
        raise LiveFactorFeatureError("official factor archive contains no observations")
    result = pd.DataFrame(rows)
    if result["period_key"].duplicated().any():
        raise LiveFactorFeatureError("official factor archive has duplicate periods")
    return result


def canonicalize_factor_archives(
    payloads: Mapping[str, bytes], *, fetched_at: datetime
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse and join FF5 and momentum at exact daily/monthly grains."""

    required = {"ff5_monthly", "ff5_daily", "momentum_monthly", "momentum_daily"}
    if set(payloads) != required:
        raise LiveFactorFeatureError("factor archive set differs from the frozen policy")
    ff5_names = ("mktrf", "smb", "hml", "rmw", "cma", "rf")
    monthly_ff5 = _parse_factor_csv(
        payloads["ff5_monthly"], date_digits=6, value_names=ff5_names
    )
    monthly_mom = _parse_factor_csv(
        payloads["momentum_monthly"], date_digits=6, value_names=("umd",)
    )
    daily_ff5 = _parse_factor_csv(
        payloads["ff5_daily"], date_digits=8, value_names=ff5_names
    )
    daily_mom = _parse_factor_csv(
        payloads["momentum_daily"], date_digits=8, value_names=("umd",)
    )
    monthly = monthly_ff5.merge(
        monthly_mom[["period_key", "umd"]], on="period_key", how="inner", validate="one_to_one"
    )
    daily = daily_ff5.merge(
        daily_mom[["period_key", "umd"]], on="period_key", how="inner", validate="one_to_one"
    )
    daily["observation_date"] = pd.to_datetime(daily["source_date"]).dt.normalize()
    last_daily = daily.groupby(daily["observation_date"].dt.to_period("M"))["observation_date"].max()
    monthly_period = pd.to_datetime(monthly["source_date"]).dt.to_period("M")
    monthly["observation_date"] = monthly_period.map(last_daily)
    if monthly["observation_date"].isna().any():
        raise LiveFactorFeatureError("monthly factors do not map to a common daily month-end")
    availability = fetched_at.astimezone(timezone.utc).date()
    for frame in (monthly, daily):
        frame["public_availability_date"] = pd.Timestamp(availability)
        frame["snapshot_fetched_at_utc"] = fetched_at.astimezone(timezone.utc).isoformat()
    monthly = monthly[["observation_date", "public_availability_date", *FACTOR_NAMES[:-1], "rf", "umd", "snapshot_fetched_at_utc"]]
    daily = daily[["observation_date", "public_availability_date", *FACTOR_NAMES[:-1], "rf", "umd", "snapshot_fetched_at_utc"]]
    return monthly.sort_values("observation_date"), daily.sort_values("observation_date")


def snapshot_official_factors(
    *, config_path: Path, output: Path, fetched_at: datetime | None = None
) -> dict[str, Any]:
    """Download one immutable, timestamped Kenneth French factor snapshot."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "research_v6_8_live_factor_policy_v1":
        raise LiveFactorFeatureError("live factor policy schema differs")
    timestamp = fetched_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise LiveFactorFeatureError("factor snapshot time must be timezone-aware")
    payloads = {
        name: _download(str(url)) for name, url in config["source_urls"].items()
    }
    monthly, daily = canonicalize_factor_archives(payloads, fetched_at=timestamp)
    raw_root = output / "raw"
    for name, payload in payloads.items():
        _atomic_bytes(raw_root / f"{name}.zip", payload)
    monthly_path = output / "monthly_factors.parquet"
    daily_path = output / "daily_factors.parquet"
    _atomic_parquet(monthly_path, monthly)
    _atomic_parquet(daily_path, daily)
    receipt = {
        "schema": "research_v6_8_official_factor_snapshot_v1",
        "status": "PASS",
        "fetched_at_utc": timestamp.astimezone(timezone.utc).isoformat(),
        "monthly_rows": int(len(monthly)),
        "daily_rows": int(len(daily)),
        "latest_monthly_observation": pd.Timestamp(monthly["observation_date"].max()).date().isoformat(),
        "latest_daily_observation": pd.Timestamp(daily["observation_date"].max()).date().isoformat(),
        "raw_sha256": {name: _sha256_bytes(payload) for name, payload in payloads.items()},
        "monthly_sha256": _sha256(monthly_path),
        "daily_sha256": _sha256(daily_path),
    }
    _atomic_json(output / "receipt.json", receipt)
    return receipt


def _signal_cutoff(signal_date: date, config: Mapping[str, Any]) -> datetime:
    cutoff_time = time.fromisoformat(str(config["signal_cutoff_local_time"]))
    zone = ZoneInfo(str(config["signal_cutoff_timezone"]))
    return datetime.combine(signal_date, cutoff_time, tzinfo=zone)


def materialize_live_factor_features(
    *,
    config_path: Path,
    factor_snapshot: Path,
    market_history_path: Path,
    market_features_path: Path,
    output: Path,
    signal_date: date,
) -> dict[str, Any]:
    """Build causal betas, idiosyncratic risk, and aggregate factor state."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    receipt = json.loads((factor_snapshot / "receipt.json").read_text(encoding="utf-8"))
    fetched_at = datetime.fromisoformat(str(receipt["fetched_at_utc"]))
    if fetched_at > _signal_cutoff(signal_date, config).astimezone(timezone.utc):
        raise LiveFactorFeatureError("factor snapshot was fetched after the signal cutoff")
    daily = pd.read_parquet(factor_snapshot / "daily_factors.parquet")
    monthly = pd.read_parquet(factor_snapshot / "monthly_factors.parquet")
    history = pd.read_parquet(market_history_path)
    market = pd.read_parquet(market_features_path)
    daily["observation_date"] = pd.to_datetime(daily["observation_date"]).dt.normalize()
    monthly["observation_date"] = pd.to_datetime(monthly["observation_date"]).dt.normalize()
    history["observation_date"] = pd.to_datetime(history["observation_date"]).dt.normalize()
    daily = daily.loc[daily["observation_date"].dt.date <= signal_date].copy()
    monthly = monthly.loc[monthly["observation_date"].dt.date <= signal_date].copy()
    if daily.empty or monthly.empty:
        raise LiveFactorFeatureError("official factor snapshot has no pre-signal data")
    latest_factor = pd.Timestamp(daily["observation_date"].max())
    age_days = (signal_date - latest_factor.date()).days
    beta = config["beta_policy"]
    factor_fresh = age_days <= int(beta["maximum_latest_factor_age_calendar_days"])
    sessions = sorted(
        pd.Timestamp(value)
        for value in history.loc[
            history["observation_date"].dt.date <= signal_date, "observation_date"
        ].dropna().unique()
    )
    if len(sessions) < int(beta["lookback_global_sessions"]):
        raise LiveFactorFeatureError("market history has too few global sessions")
    lookback = sessions[-int(beta["lookback_global_sessions"]):]
    age_by_date = {day: len(lookback) - 1 - index for index, day in enumerate(lookback)}
    factors = daily.set_index("observation_date")[[*FACTOR_NAMES, "rf"]]
    if market["live_security_id"].duplicated().any():
        raise LiveFactorFeatureError("market features duplicate live security identities")
    market_ids = market["live_security_id"].astype(str).tolist()
    history_by_security = {
        str(security_id): group
        for security_id, group in history.groupby("live_security_id", sort=False)
    }
    rows: list[dict[str, Any]] = []
    for security_id in market_ids:
        group = history_by_security.get(security_id)
        if group is None:
            row = {"live_security_id": security_id}
            for name in BETA_NAMES:
                row[name] = np.nan
            row["beta_label_idiosyncratic_volatility"] = np.nan
            rows.append(row)
            continue
        security = group.set_index("observation_date")["total_return"]
        dates = [day for day in lookback if day in security.index and day in factors.index]
        fit = None
        if factor_fresh and len(dates) >= int(beta["minimum_valid_observations"]):
            factor_rows = factors.loc[dates]
            fit = ewls_factor_fit(
                pd.to_numeric(security.loc[dates], errors="coerce").to_numpy(),
                pd.to_numeric(factor_rows["rf"], errors="coerce").to_numpy(),
                factor_rows[list(FACTOR_NAMES)].apply(pd.to_numeric, errors="coerce").to_numpy(),
                [age_by_date[day] for day in dates],
                half_life_sessions=float(beta["half_life_sessions"]),
                minimum_observations=int(beta["minimum_valid_observations"]),
            )
        row: dict[str, Any] = {"live_security_id": security_id}
        for name, value in zip(BETA_NAMES, () if fit is None else fit.slopes):
            row[name] = value
        if fit is None:
            for name in BETA_NAMES:
                row[name] = np.nan
            row["beta_label_idiosyncratic_volatility"] = np.nan
        else:
            row["beta_label_idiosyncratic_volatility"] = fit.annualized_idiosyncratic_volatility
        rows.append(row)
    result = pd.DataFrame(rows)
    eligible_ids = set(
        market.loc[market["long_eligible"].astype(bool), "live_security_id"].astype(str)
    )
    for name in BETA_NAMES:
        population = pd.to_numeric(
            result.loc[result["live_security_id"].isin(eligible_ids), name], errors="coerce"
        ).dropna()
        if not population.empty:
            lower = float(population.quantile(float(beta["cross_sectional_clip_lower"])))
            upper = float(population.quantile(float(beta["cross_sectional_clip_upper"])))
            result[name] = pd.to_numeric(result[name], errors="coerce").clip(lower, upper)
    signal_period = pd.Period(signal_date, freq="M")
    official_period = pd.Timestamp(monthly["observation_date"].max()).to_period("M")
    exact_signal_month_available = official_period == signal_period
    latest_rf = float(pd.to_numeric(monthly.sort_values("observation_date").iloc[-1]["rf"]))
    for name in TRAILING_FACTOR_NAMES:
        result[name] = np.nan
    result["risk_free_rate_1m"] = latest_rf
    result.insert(1, "signal_date", pd.Timestamp(signal_date))
    result.insert(2, "latest_official_factor_date", latest_factor)
    result.insert(3, "latest_factor_age_calendar_days", int(age_days))
    result.insert(4, "exact_signal_month_factors_available", bool(exact_signal_month_available))
    if (
        result["live_security_id"].duplicated().any()
        or len(result) != len(market)
        or result["live_security_id"].tolist() != market_ids
    ):
        raise LiveFactorFeatureError("factor feature grain/order differs from market features")
    output_path = output / "factor_features.parquet"
    _atomic_parquet(output_path, result)
    feature_receipt = {
        "schema": "research_v6_8_live_factor_features_receipt_v1",
        "status": "PASS",
        "signal_date": signal_date.isoformat(),
        "rows": int(len(result)),
        "eligible_rows": int(len(eligible_ids)),
        "latest_official_factor_date": latest_factor.date().isoformat(),
        "latest_factor_age_calendar_days": int(age_days),
        "factor_fresh_for_beta_policy": bool(factor_fresh),
        "exact_signal_month_factors_available": bool(exact_signal_month_available),
        "beta_complete_share": float(result[list(BETA_NAMES)].notna().all(axis=1).mean()),
        "trailing_factor_fields_intentionally_missing": list(TRAILING_FACTOR_NAMES),
        "risk_free_rate_1m": latest_rf,
        "output_sha256": _sha256(output_path),
    }
    _atomic_json(output / "receipt.json", feature_receipt)
    return feature_receipt


__all__ = [
    "BETA_NAMES",
    "FACTOR_NAMES",
    "TRAILING_FACTOR_NAMES",
    "LiveFactorFeatureError",
    "canonicalize_factor_archives",
    "materialize_live_factor_features",
    "snapshot_official_factors",
]
