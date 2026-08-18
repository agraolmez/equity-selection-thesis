#!/usr/bin/env python3
"""Run six bounded V6.3 signal-to-weight challengers against the 1.5% incumbent."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_v6_1_final_optimizer_challenge as inference_tools  # noqa: E402
from scripts import run_v6_1_portfolio_step_02 as base  # noqa: E402
from portfolio_bot.research_v6.models import rank_within_month  # noqa: E402
from portfolio_bot.research_v6_1.portfolio import (  # noqa: E402
    FACTOR_NAMES,
    solve_expected_net_return_portfolio,
)
from portfolio_bot.research_v6_1.runtime import _run_xgboost  # noqa: E402
from portfolio_bot.research_v6_3.portfolio import (  # noqa: E402
    causal_opportunity_percentiles,
    latent_residual_risk,
    opportunity_multiplier,
)
from portfolio_bot.research_v6_3.signals import (  # noqa: E402
    KEYS,
    TARGET_COLUMNS,
    add_confidence_diagnostics,
    build_three_month_target,
    causal_specialist_mixture,
    compose_multitask_scores,
    feature_expert_indices,
    feature_group_coverage,
    fit_multitask_selector,
    predict_multitask_selector,
)


DEFAULT_CONFIG = ROOT / "configs/research_v6_3/signal_weight_research.json"
DEFAULT_OUTPUT = ROOT / "artifacts/research_v6_3/signal_weight_research"
STABLE_PYTHON = Path("/opt/anaconda3/envs/finance/bin/python")
POLICY = "return_plus_ff5_momentum"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _code_identity() -> str:
    files = [
        Path(__file__).resolve(),
        ROOT / "src/portfolio_bot/research_v6_3/signals.py",
        ROOT / "src/portfolio_bot/research_v6_3/portfolio.py",
        ROOT / "src/portfolio_bot/research_v6_1/portfolio.py",
        ROOT / "scripts/run_v6_1_portfolio_step_02.py",
        ROOT / "src/portfolio_bot/allocation/costs.py",
        ROOT / "src/portfolio_bot/risk_cost/cost_math.py",
    ]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _verify(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path], str]:
    study = json.loads(config_path.read_text())
    if study.get("status") != "AUTHORIZED_READY_TO_RUN":
        raise RuntimeError("V6.3 signal-weight contract is not authorized")
    base_path = ROOT / study["base_portfolio_contract"]["path"]
    if _sha256(base_path) != study["base_portfolio_contract"]["sha256"]:
        raise RuntimeError("bound V6.1 portfolio contract changed")
    incumbent_manifest = ROOT / study["incumbent"]["frontier_root"] / "run_manifest.json"
    if _sha256(incumbent_manifest) != study["incumbent"]["frontier_manifest_sha256"]:
        raise RuntimeError("bound incumbent frontier manifest changed")
    cache_root = ROOT / study["model_input_cache"]["root"]
    cache_paths = {
        "rows": cache_root / "rows.parquet",
        "matrix": cache_root / "encoded_values.npy",
        "metadata": cache_root / "feature_metadata.json",
        "views": ROOT / study["feature_views"]["path"],
    }
    expected = {
        "rows": study["model_input_cache"]["rows_sha256"],
        "matrix": study["model_input_cache"]["matrix_sha256"],
        "metadata": study["model_input_cache"]["metadata_sha256"],
        "views": study["feature_views"]["sha256"],
    }
    for name, path in cache_paths.items():
        if not path.is_file() or _sha256(path) != expected[name]:
            raise RuntimeError(f"bound V6.3 model input changed: {name}")
    frozen, _, input_identity, paths = base.verify_contract(base_path)
    identity = _canonical(
        {
            "config": _sha256(config_path),
            "base_inputs": input_identity,
            "cache": expected,
            "incumbent": study["incumbent"]["frontier_manifest_sha256"],
        }
    )
    return study, frozen, {**paths, **cache_paths}, identity


def _official_prediction_path(target: str) -> Path:
    return (
        ROOT
        / "artifacts/research_v6_1/selector_step_01/481e306553c2-16046ee58521-d5dadd9cd9c3"
        / target
        / "official_predictions.parquet"
    )


def _confidence_signals(output: Path) -> Path:
    destination = output / "signals/confidence.parquet"
    if destination.is_file():
        return destination
    specifications = {
        "long": (
            "long_net",
            "ensemble__cam_free_mlp_plus_cam__equal",
            ["cam_free__mlp", "cam_direct"],
        ),
        "short": (
            "short_net",
            "ensemble__correlation_reduced_xgb_mlp__equal",
            ["correlation_reduced__xgboost", "correlation_reduced__mlp"],
        ),
        "ff5": (
            "ff5_momentum",
            "ensemble__cam_free_mlp_plus_cam__equal",
            ["cam_free__mlp", "cam_direct"],
        ),
    }
    result: pd.DataFrame | None = None
    for prefix, (target, base_column, components) in specifications.items():
        frame = pd.read_parquet(
            _official_prediction_path(target),
            columns=[*KEYS, base_column, *components],
        )
        frame["holding_month"] = frame["holding_month"].astype(str)
        frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise")
        frame = add_confidence_diagnostics(
            frame,
            base_column=base_column,
            component_columns=components,
            prefix=prefix,
        )
        keep = [*KEYS, f"{prefix}_uncertainty"]
        result = frame[keep] if result is None else result.merge(
            frame[keep], on=KEYS, how="inner", validate="one_to_one"
        )
    assert result is not None
    _atomic_parquet(destination, result)
    return destination


def _annual_blocks(rows: pd.DataFrame, start: str, end: str) -> list[tuple[str, list[str]]]:
    months = sorted(
        rows.loc[rows["holding_month"].astype(str).between(start, end), "holding_month"]
        .astype(str)
        .unique()
    )
    blocks: list[tuple[str, list[str]]] = []
    for year in sorted({month[:4] for month in months}):
        block = [month for month in months if month.startswith(year)]
        blocks.append((block[0], block))
    return blocks


def _specialist_predictions(
    study: Mapping[str, Any],
    output: Path,
    rows: pd.DataFrame,
    matrix: np.ndarray,
    metadata: Mapping[str, Any],
    run_identity: str,
    *,
    smoke: bool,
) -> Path:
    destination = output / "signals/specialist_scores.parquet"
    if destination.is_file():
        return destination
    specification = study["ideas"]["03_specialist_mixture"]
    expert_indices = feature_expert_indices(metadata, specification["experts"])
    coverage = feature_group_coverage(matrix, metadata, expert_indices)
    blocks = _annual_blocks(
        rows, study["evidence"]["signal_start"], study["evidence"]["historical_test_end"]
    )
    if smoke:
        blocks = blocks[:1]
    months = rows["holding_month"].astype(str).to_numpy()
    available = pd.to_datetime(rows["label_available_date"], errors="coerce")
    signal_rows = rows.loc[
        rows["holding_month"].astype(str).between(
            study["evidence"]["signal_start"],
            (blocks[-1][1][-1] if smoke else study["evidence"]["historical_test_end"]),
        )
    ].copy()
    component_by_target: dict[str, pd.DataFrame] = {}
    parameters = dict(specification["xgboost"])
    for target, target_column in TARGET_COLUMNS.items():
        target_values = pd.to_numeric(rows[target_column], errors="coerce").to_numpy(float)
        components = signal_rows[KEYS].copy()
        for expert, indices in expert_indices.items():
            parts: list[pd.DataFrame] = []
            task_root = output / "signals/specialists" / target / expert
            for refit_month, block_months in blocks:
                prediction_path = task_root / "checkpoints" / f"{refit_month}.parquet"
                receipt_path = task_root / "checkpoints" / f"{refit_month}.json"
                block_identity = _canonical(
                    {
                        "run": run_identity,
                        "target": target,
                        "expert": expert,
                        "refit": refit_month,
                        "months": block_months,
                        "features": indices.tolist(),
                        "parameters": parameters,
                    }
                )
                if prediction_path.is_file() and receipt_path.is_file():
                    receipt = json.loads(receipt_path.read_text())
                    if receipt.get("block_identity") != block_identity:
                        raise RuntimeError(f"specialist checkpoint mismatch: {receipt_path}")
                    parts.append(pd.read_parquet(prediction_path))
                    continue
                signal_dates = pd.to_datetime(
                    rows.loc[rows["holding_month"].astype(str).eq(refit_month), "signal_date"],
                    errors="raise",
                ).unique()
                if len(signal_dates) != 1:
                    raise RuntimeError(f"specialist refit has invalid signal date: {refit_month}")
                train = np.flatnonzero(
                    (months < refit_month)
                    & available.le(pd.Timestamp(signal_dates[0])).to_numpy(bool)
                    & np.isfinite(target_values)
                ).astype(np.int64)
                if smoke:
                    train = train[-min(len(train), 20_000) :]
                forecast = np.flatnonzero(np.isin(months, block_months)).astype(np.int64)
                effective_parameters = dict(parameters)
                if smoke:
                    effective_parameters["n_estimators"] = 20
                raw, _, _ = _run_xgboost(
                    matrix=matrix,
                    feature_index=indices,
                    train_index=train,
                    forecast_index=forecast,
                    target=target_values,
                    months=months,
                    objective="continuous_rank",
                    parameters=effective_parameters,
                    seed=20260814,
                )
                part = rows.iloc[forecast][KEYS].copy()
                part[expert] = rank_within_month(raw, part["holding_month"])
                _atomic_parquet(prediction_path, part)
                _atomic_json(
                    receipt_path,
                    {
                        "schema": "research_v6_3_specialist_checkpoint_v1",
                        "block_identity": block_identity,
                        "train_rows": len(train),
                        "forecast_rows": len(forecast),
                        "forecast_months": block_months,
                    },
                )
                parts.append(part)
                print(f"SIGNAL specialist {target} {expert} {refit_month} complete", flush=True)
            expert_frame = pd.concat(parts, ignore_index=True)
            components = components.merge(
                expert_frame, on=KEYS, how="left", validate="one_to_one"
            )
        component_by_target[target] = components

    final = signal_rows[KEYS].copy()
    for temperature in specification["temperatures"]:
        token = str(temperature).replace(".", "p")
        for target, target_column in TARGET_COLUMNS.items():
            components = component_by_target[target].copy()
            # Reorder the coverage arrays to the prediction frame explicitly.
            row_lookup = rows.reset_index().set_index(["holding_month", "permno"])["index"]
            aligned_index = components.set_index(["holding_month", "permno"]).index.map(row_lookup)
            aligned_coverage = {
                expert: values[np.asarray(aligned_index, dtype=int)]
                for expert, values in coverage.items()
            }
            mixed = causal_specialist_mixture(
                components,
                signal_rows,
                aligned_coverage,
                components=list(specification["experts"]),
                target_column=target_column,
                temperature=float(temperature),
                minimum_weight=float(specification["minimum_component_weight"]),
                history_months=int(specification["quality_history_months"]),
            )
            column = f"specialist_{target}_t{token}"
            final = final.merge(
                mixed[[*KEYS, "score_rank"]].rename(columns={"score_rank": column}),
                on=KEYS,
                how="left",
                validate="one_to_one",
            )
            weights = mixed.attrs.get("monthly_weights")
            if isinstance(weights, pd.DataFrame):
                _atomic_parquet(
                    output / "signals" / f"specialist_weights_{target}_t{token}.parquet",
                    weights,
                )
    _atomic_parquet(destination, final)
    return destination


def _multitask_predictions(
    study: Mapping[str, Any],
    output: Path,
    rows: pd.DataFrame,
    matrix: np.ndarray,
    run_identity: str,
    *,
    smoke: bool,
) -> Path:
    destination = output / "signals/multitask_predictions.parquet"
    if destination.is_file():
        return destination
    specification = study["ideas"]["04_multitask_multihorizon"]
    enriched = rows.merge(
        build_three_month_target(rows),
        on=["holding_month", "permno"],
        how="left",
        validate="one_to_one",
    )
    target_columns = [
        "target_rank_long_net",
        "target_rank_short_net",
        "target_rank_ff5_momentum",
        "target_rank_return_3m",
    ]
    target_availability = [
        "label_available_date",
        "label_available_date",
        "label_available_date",
        "target_available_date_3m",
    ]
    targets = enriched[target_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    availability = np.column_stack(
        [
            pd.to_datetime(enriched[column], errors="coerce").to_numpy(
                dtype="datetime64[ns]"
            )
            for column in target_availability
        ]
    )
    months = enriched["holding_month"].astype(str).to_numpy()
    blocks = _annual_blocks(
        enriched,
        study["evidence"]["signal_start"],
        study["evidence"]["historical_test_end"],
    )
    if smoke:
        blocks = blocks[:1]
    parts: list[pd.DataFrame] = []
    for refit_month, block_months in blocks:
        checkpoint = output / "signals/multitask/checkpoints" / f"{refit_month}.parquet"
        receipt_path = output / "signals/multitask/checkpoints" / f"{refit_month}.json"
        block_identity = _canonical(
            {
                "run": run_identity,
                "refit": refit_month,
                "months": block_months,
                "specification": specification,
                "features": "all_850_encoded",
            }
        )
        if checkpoint.is_file() and receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("block_identity") != block_identity:
                raise RuntimeError(f"multitask checkpoint mismatch: {receipt_path}")
            parts.append(pd.read_parquet(checkpoint))
            continue
        signal_dates = pd.to_datetime(
            enriched.loc[enriched["holding_month"].astype(str).eq(refit_month), "signal_date"],
            errors="raise",
        ).unique()
        if len(signal_dates) != 1:
            raise RuntimeError(f"multitask refit has invalid signal date: {refit_month}")
        signal_date = np.datetime64(pd.Timestamp(signal_dates[0]).to_datetime64())
        allowed = (months < refit_month)[:, None] & (availability <= signal_date)
        train_targets = targets.copy()
        train_targets[~allowed] = np.nan
        train = np.flatnonzero(np.isfinite(train_targets).any(axis=1)).astype(np.int64)
        if smoke:
            train = train[-min(len(train), 20_000) :]
        forecast = np.flatnonzero(np.isin(months, block_months)).astype(np.int64)
        model = fit_multitask_selector(
            np.asarray(matrix[train], dtype=np.float32),
            train_targets[train],
            seed=int(specification["seed"]),
            hidden_layers=specification["hidden_layers"],
            dropout=float(specification["dropout"]),
            epochs=(1 if smoke else int(specification["epochs"])),
            batch_size=int(specification["batch_size"]),
            learning_rate=float(specification["learning_rate"]),
            weight_decay=float(specification["weight_decay"]),
        )
        prediction = predict_multitask_selector(
            model, np.asarray(matrix[forecast], dtype=np.float32)
        )
        part = enriched.iloc[forecast][KEYS].copy()
        for index, name in enumerate(["long", "short", "ff5", "3m"]):
            part[f"rank_head_{name}"] = prediction.ranks[:, index]
        for index, name in enumerate(["long", "short", "ff5"]):
            part[f"tail_probability_{name}"] = prediction.tail_probabilities[:, index]
        _atomic_parquet(checkpoint, part)
        _atomic_json(
            receipt_path,
            {
                "schema": "research_v6_3_multitask_checkpoint_v1",
                "block_identity": block_identity,
                "train_rows": len(train),
                "forecast_rows": len(forecast),
                "forecast_months": block_months,
            },
        )
        parts.append(part)
        print(f"SIGNAL multitask {refit_month} complete", flush=True)
    combined = pd.concat(parts, ignore_index=True).sort_values(
        ["holding_month", "permno"], kind="mergesort"
    )
    _atomic_parquet(destination, combined)
    return destination


def _prepare_signals(
    study: Mapping[str, Any],
    paths: Mapping[str, Path],
    output: Path,
    run_identity: str,
    *,
    smoke: bool,
) -> dict[str, Path]:
    rows = pd.read_parquet(paths["rows"])
    rows["holding_month"] = rows["holding_month"].astype(str)
    rows["signal_date"] = pd.to_datetime(rows["signal_date"], errors="raise")
    rows["label_available_date"] = pd.to_datetime(
        rows["label_available_date"], errors="coerce"
    )
    matrix = np.load(paths["matrix"], mmap_mode="r")
    metadata = json.loads(paths["metadata"].read_text())
    if matrix.shape != (len(rows), len(metadata["feature_names"])):
        raise RuntimeError("V6.3 model matrix and row cache are not aligned")
    confidence = _confidence_signals(output)
    specialist = _specialist_predictions(
        study, output, rows, matrix, metadata, run_identity, smoke=smoke
    )
    multitask = _multitask_predictions(
        study, output, rows, matrix, run_identity, smoke=smoke
    )
    manifest = {
        "schema": "research_v6_3_signal_artifacts_v1",
        "status": "COMPLETE",
        "run_identity": run_identity,
        "confidence": str(confidence.relative_to(output)),
        "specialist": str(specialist.relative_to(output)),
        "multitask": str(multitask.relative_to(output)),
        "uses_all_encoded_features": True,
        "outcomes_used_only_for_causal_training_or_trailing_quality": True,
    }
    _atomic_json(output / "signals/manifest.json", manifest)
    return {"confidence": confidence, "specialist": specialist, "multitask": multitask}


def _replace_scores(
    inputs: base.PreparedInputs,
    scores: pd.DataFrame,
    columns: Mapping[str, str],
) -> base.PreparedInputs:
    forecast = inputs.forecasts.copy()
    incoming_columns: dict[str, str] = {}
    for target, source in columns.items():
        destination = f"score_{target}"
        incoming_columns[source] = (
            f"__v6_3_{target}" if source == destination else source
        )
    incoming = scores[[*KEYS, *columns.values()]].copy().rename(
        columns=incoming_columns
    )
    forecast = forecast.merge(incoming, on=KEYS, how="left", validate="one_to_one")
    for target, source in columns.items():
        destination = f"score_{target}"
        source_column = incoming_columns[source]
        available = pd.to_numeric(forecast[source_column], errors="coerce")
        forecast[destination] = available.fillna(forecast[destination])
        forecast[f"rank_{target}"] = forecast.groupby(
            "holding_month", observed=True, sort=False
        )[destination].transform(lambda values: base.percentile_rank(values))
    return base.PreparedInputs(
        forecasts=forecast,
        risk=inputs.risk,
        covariance=inputs.covariance,
        factors=inputs.factors,
        benchmarks=inputs.benchmarks,
    )


def _input_variants(
    study: Mapping[str, Any],
    paths: Mapping[str, Path],
    signals: Mapping[str, Path],
    frozen: dict[str, Any],
) -> tuple[dict[str, base.PreparedInputs], dict[str, Any]]:
    loader = copy.deepcopy(frozen)
    loader["evidence"]["portfolio_start"] = study["evidence"]["signal_start"]
    loader["evidence"]["portfolio_end"] = study["evidence"]["historical_test_end"]
    loader["optimizer"]["maximum_position_weight"] = float(
        study["fixed_portfolio_policy"]["maximum_position_weight"]
    )
    inputs = base.load_inputs(loader, paths)
    confidence = pd.read_parquet(signals["confidence"])
    inputs.forecasts = inputs.forecasts.merge(
        confidence, on=KEYS, how="left", validate="one_to_one"
    )
    variants: dict[str, base.PreparedInputs] = {"incumbent": inputs}

    specialist = pd.read_parquet(signals["specialist"])
    for temperature in study["ideas"]["03_specialist_mixture"]["temperatures"]:
        token = str(temperature).replace(".", "p")
        variants[f"specialist_t{token}"] = _replace_scores(
            inputs,
            specialist,
            {
                "long_net": f"specialist_long_net_t{token}",
                "short_net": f"specialist_short_net_t{token}",
                "ff5_momentum": f"specialist_ff5_momentum_t{token}",
            },
        )

    multitask_raw = pd.read_parquet(signals["multitask"])
    all_blends = sorted(
        set(study["ideas"]["04_multitask_multihorizon"]["horizon_blends"])
        | set(study["ideas"]["05_decision_aware_rank_mask"]["horizon_blends"])
    )
    for blend in all_blends:
        token = str(blend).replace(".", "p")
        composed = compose_multitask_scores(multitask_raw, horizon_blend=float(blend))
        variant = _replace_scores(
            inputs,
            composed,
            {
                "long_net": "score_long_net",
                "short_net": "score_short_net",
                "ff5_momentum": "score_ff5_momentum",
            },
        )
        probabilities = composed[
            [*KEYS, "tail_probability_long", "tail_probability_short", "tail_probability_ff5"]
        ]
        variant.forecasts = variant.forecasts.merge(
            probabilities, on=KEYS, how="left", validate="one_to_one"
        )
        variants[f"multitask_b{token}"] = variant
    return variants, loader


def _profiles(study: Mapping[str, Any]) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = [
        {"id": "incumbent_1p5", "idea": "control", "signal_variant": "incumbent"}
    ]
    for penalty in study["ideas"]["01_confidence_adjusted_edge"]["penalties"]:
        token = str(penalty).replace(".", "p")
        profiles.append(
            {
                "id": f"confidence_p{token}",
                "idea": "01_confidence_adjusted_edge",
                "signal_variant": "incumbent",
                "confidence_penalty": float(penalty),
            }
        )
    for strength in study["ideas"]["02_opportunity_breadth_scaling"]["strengths"]:
        token = str(strength).replace(".", "p")
        profiles.append(
            {
                "id": f"breadth_s{token}",
                "idea": "02_opportunity_breadth_scaling",
                "signal_variant": "incumbent",
                "confidence_penalty": 1.0,
                "breadth_strength": float(strength),
            }
        )
    for temperature in study["ideas"]["03_specialist_mixture"]["temperatures"]:
        token = str(temperature).replace(".", "p")
        profiles.append(
            {
                "id": f"specialist_t{token}",
                "idea": "03_specialist_mixture",
                "signal_variant": f"specialist_t{token}",
            }
        )
    for blend in study["ideas"]["04_multitask_multihorizon"]["horizon_blends"]:
        token = str(blend).replace(".", "p")
        profiles.append(
            {
                "id": f"multitask_b{token}",
                "idea": "04_multitask_multihorizon",
                "signal_variant": f"multitask_b{token}",
            }
        )
    for blend in study["ideas"]["05_decision_aware_rank_mask"]["horizon_blends"]:
        blend_token = str(blend).replace(".", "p")
        for threshold in study["ideas"]["05_decision_aware_rank_mask"][
            "tail_probability_thresholds"
        ]:
            threshold_token = str(threshold).replace(".", "p")
            profiles.append(
                {
                    "id": f"decision_b{blend_token}_q{threshold_token}",
                    "idea": "05_decision_aware_rank_mask",
                    "signal_variant": f"multitask_b{blend_token}",
                    "tail_probability_threshold": float(threshold),
                }
            )
    for item in study["ideas"]["06_latent_risk_layer"]["profiles"]:
        share = str(item["latent_variance_share"]).replace(".", "p")
        profiles.append(
            {
                "id": f"latent_k{item['latent_factors']}_s{share}",
                "idea": "06_latent_risk_layer",
                "signal_variant": "incumbent",
                **item,
            }
        )
    return profiles


def _opportunity_map(inputs: base.PreparedInputs) -> dict[str, float]:
    frame = inputs.forecasts[
        [
            "holding_month",
            "permno",
            "rank_long_net",
            "rank_short_net",
            "long_uncertainty",
            "short_uncertainty",
        ]
    ].merge(
        inputs.risk[["holding_month", "permno", "long_eligible", "short_eligible"]],
        on=["holding_month", "permno"],
        how="inner",
        validate="one_to_one",
    )
    records: list[tuple[str, float]] = []
    for month, group in frame.groupby("holding_month", observed=True, sort=True):
        long = group["long_eligible"].fillna(False).to_numpy(bool) & group[
            "rank_long_net"
        ].ge(0.80).to_numpy(bool)
        short = group["short_eligible"].fillna(False).to_numpy(bool) & group[
            "rank_short_net"
        ].ge(0.80).to_numpy(bool)
        strength: list[np.ndarray] = []
        if long.any():
            strength.append(
                (group.loc[long, "rank_long_net"].to_numpy(float) - 0.80)
                * (1.0 - group.loc[long, "long_uncertainty"].to_numpy(float))
            )
        if short.any():
            strength.append(
                (group.loc[short, "rank_short_net"].to_numpy(float) - 0.80)
                * (1.0 - group.loc[short, "short_uncertainty"].to_numpy(float))
            )
        score = float(np.mean(np.concatenate(strength))) if strength else 0.0
        records.append((str(month), score))
    monthly = pd.Series(
        [value for _, value in records],
        index=[month for month, _ in records],
        dtype=float,
    )
    percentile = causal_opportunity_percentiles(monthly, minimum_history_months=12)
    return {str(month): float(value) for month, value in percentile.items()}


def _latent_history(inputs: base.PreparedInputs) -> pd.DataFrame:
    factor_columns = [f"beta_risk_{name}" for name in FACTOR_NAMES]
    history = inputs.forecasts[
        ["holding_month", "permno", "stock_excess_return", "label_available_date"]
    ].merge(
        inputs.risk[["holding_month", "permno", *factor_columns]],
        on=["holding_month", "permno"],
        how="inner",
        validate="one_to_one",
    )
    history = history.merge(
        inputs.factors[["holding_month", *FACTOR_NAMES]],
        on="holding_month",
        how="left",
        validate="many_to_one",
    )
    return history


def _solve_profile_month(
    *,
    study: Mapping[str, Any],
    frame: pd.DataFrame,
    covariance_root: np.ndarray,
    loader: dict[str, Any],
    profile: Mapping[str, Any],
    previous_map: dict[int, float],
    nav_dollars: float,
    opportunity_percentile: float,
    latent_history: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if profile["idea"] == "control":
        variant = {
            "id": str(profile["id"]),
            "policy": POLICY,
            "kind": "optimized",
            "method": "optimized",
            "volatility": float(loader["optimizer"]["primary_forecast_volatility"]),
        }
        solution, positions, summary = base._solve_variant_month(
            frame, covariance_root, loader, variant, previous_map, nav_dollars
        )
        previous = base._aligned_previous(frame, previous_map)
        participation = np.divide(
            np.abs(solution.weights - previous) * nav_dollars,
            frame["adv20_usd"].to_numpy(float),
            out=np.zeros(len(frame)),
            where=frame["adv20_usd"].to_numpy(float) > 0.0,
        )
        positions["adv_participation"] = positions["permno"].astype(int).map(
            dict(zip(frame["permno"].astype(int), participation, strict=True))
        )
        summary.update(
            {
                "profile": str(profile["id"]),
                "idea": "control",
                "maximum_adv_participation": float(np.max(participation, initial=0.0)),
                "latent_factors": 0,
                "latent_explained_residual_variance": 0.0,
                "expected_return_scale": 1.0,
            }
        )
        return positions, summary

    long_mask, short_mask = base._policy_masks(frame, loader, POLICY)
    if profile["idea"] == "05_decision_aware_rank_mask":
        threshold = float(profile["tail_probability_threshold"])
        long_mask &= pd.to_numeric(
            frame["tail_probability_long"], errors="coerce"
        ).fillna(0.0).ge(threshold).to_numpy(bool)
        short_mask &= pd.to_numeric(
            frame["tail_probability_short"], errors="coerce"
        ).fillna(0.0).ge(threshold).to_numpy(bool)
    previous = base._aligned_previous(frame, previous_map)
    opposing = ((previous > 0.0) & short_mask) | ((previous < 0.0) & long_mask)
    long_mask[opposing] = False
    short_mask[opposing] = False
    reduction_only = (previous != 0.0) & ~(long_mask | short_mask)

    mu = np.zeros(len(frame), dtype=np.float64)
    mu[long_mask] = frame.loc[long_mask, "expected_long"].to_numpy(float)
    mu[short_mask] = -frame.loc[short_mask, "expected_short"].to_numpy(float)
    penalty = float(profile.get("confidence_penalty", 0.0))
    if penalty:
        uncertainty = np.zeros(len(frame), dtype=float)
        uncertainty[long_mask] = pd.to_numeric(
            frame.loc[long_mask, "long_uncertainty"], errors="coerce"
        ).fillna(0.5)
        uncertainty[short_mask] = pd.to_numeric(
            frame.loc[short_mask, "short_uncertainty"], errors="coerce"
        ).fillna(0.5)
        mu *= np.exp(-penalty * uncertainty)
    expected_scale = 1.0
    if "breadth_strength" in profile:
        breadth = study["ideas"]["02_opportunity_breadth_scaling"]
        expected_scale = opportunity_multiplier(
            opportunity_percentile,
            strength=float(profile["breadth_strength"]),
            minimum=float(breadth["minimum_multiplier"]),
            maximum=float(breadth["maximum_multiplier"]),
        )
        mu *= expected_scale

    factor_columns = [f"beta_risk_{name}" for name in FACTOR_NAMES]
    risk_betas = frame[factor_columns].to_numpy(float)
    risk_root = covariance_root
    specific = frame["specific_variance_monthly"].to_numpy(float)
    latent_count = 0
    latent_explained = 0.0
    if profile["idea"] == "06_latent_risk_layer":
        latent_spec = study["ideas"]["06_latent_risk_layer"]
        latent = latent_residual_risk(
            frame,
            latent_history,
            covariance_root,
            factor_columns=factor_columns,
            latent_factors=int(profile["latent_factors"]),
            latent_variance_share=float(profile["latent_variance_share"]),
            trailing_months=int(latent_spec["trailing_months"]),
            minimum_months=int(latent_spec["minimum_months"]),
            specific_variance_floor_fraction=float(
                latent_spec["specific_variance_floor_fraction"]
            ),
        )
        risk_betas = latent.factor_betas
        risk_root = latent.covariance_root
        specific = latent.specific_variance_monthly
        latent_count = latent.factors
        latent_explained = latent.explained_residual_variance

    optimizer = loader["optimizer"]
    primary = loader["costs"]["primary"]
    solution = solve_expected_net_return_portfolio(
        expected_stock_excess_return=mu,
        long_candidate=long_mask,
        short_candidate=short_mask,
        reduction_only=reduction_only,
        previous_weights=previous,
        factor_betas=risk_betas,
        factor_covariance_root=risk_root,
        specific_variance_monthly=specific,
        full_spread=frame["full_spread_20d"].to_numpy(float),
        adv20_usd=frame["adv20_usd"].to_numpy(float),
        impact_volatility_daily=frame["impact_volatility_daily"].to_numpy(float),
        borrow_rate_annual=frame[str(primary["borrow_column"])].to_numpy(float),
        sic2_industry=frame["sic2_industry_category"].to_numpy(str),
        nav_dollars=nav_dollars,
        forecast_volatility_target=float(optimizer["primary_forecast_volatility"]),
        maximum_gross=float(optimizer["maximum_gross"]),
        minimum_long_gross=float(optimizer["minimum_long_gross"]),
        minimum_short_gross=float(optimizer["minimum_short_gross"]),
        minimum_net=float(optimizer["minimum_net"]),
        maximum_net=float(optimizer["maximum_net"]),
        maximum_position_weight=float(optimizer["maximum_position_weight"]),
        maximum_one_way_turnover=float(optimizer["maximum_one_way_turnover"]),
        maximum_one_way_adv_participation=float(
            optimizer["maximum_one_way_adv_participation"]
        ),
        market_beta_absolute_limit=float(optimizer["market_beta_absolute_limit"]),
        other_factor_absolute_limit=float(optimizer["other_factor_absolute_limit"]),
        sic2_industry_net_absolute_limit=float(
            optimizer["sic2_industry_net_absolute_limit"]
        ),
        sic2_industry_gross_limit=float(optimizer["sic2_industry_gross_limit"]),
        spread_multiplier=float(primary["spread_multiplier"]),
        impact_coefficient=float(primary["impact_coefficient"]),
        financing_rate_annual=float(primary["financing_rate_annual"]),
        solver=str(optimizer["solver"]),
    )
    weights = solution.weights
    primary_cost, stress_cost = base._cost_breakdowns(
        weights, previous, frame, loader, nav_dollars
    )
    excess = frame["stock_excess_return"].to_numpy(float)
    if np.any((weights != 0.0) & ~np.isfinite(excess)):
        raise RuntimeError("a V6.3 holding lacks an authenticated outcome")
    excess = np.where(np.isfinite(excess), excess, 0.0)
    gross_excess = float(weights @ excess)
    rf = float(frame["rf"].iloc[0])
    net_return = rf + gross_excess - primary_cost.total
    stress_return = rf + gross_excess - stress_cost.total
    if 1.0 + net_return <= 0.0:
        raise RuntimeError("V6.3 portfolio NAV became nonpositive")
    end_drift = weights * (1.0 + excess + rf) / (1.0 + net_return)
    participation = np.divide(
        np.abs(weights - previous) * nav_dollars,
        frame["adv20_usd"].to_numpy(float),
        out=np.zeros(len(frame)),
        where=frame["adv20_usd"].to_numpy(float) > 0.0,
    )
    positions = frame[[*KEYS, "sic2_industry_category", "return_source"]].copy()
    positions["weight"] = weights
    positions["previous_weight"] = previous
    positions["end_drift_weight"] = end_drift
    positions["stock_excess_return"] = excess
    positions["long_candidate"] = long_mask
    positions["short_candidate"] = short_mask
    positions["expected_stock_excess_return"] = mu
    positions["adv_participation"] = participation
    positions = positions.loc[
        positions["weight"].ne(0.0)
        | positions["previous_weight"].ne(0.0)
        | positions["long_candidate"]
        | positions["short_candidate"]
    ].reset_index(drop=True)
    summary = {
        "holding_month": str(frame["holding_month"].iloc[0]),
        "signal_date": str(pd.Timestamp(frame["signal_date"].iloc[0]).date()),
        # Keep the frozen V6.1 aggregation contract intact. In V6.3 the
        # profile identifier is the portfolio variant identifier.
        "variant": str(profile["id"]),
        "policy": POLICY,
        "weighting": "optimized",
        "volatility_target": float(optimizer["primary_forecast_volatility"]),
        "profile": str(profile["id"]),
        "idea": str(profile["idea"]),
        "solver_status": solution.status,
        "nav_start": float(nav_dollars),
        "nav_end_primary": float(nav_dollars * (1.0 + net_return)),
        "rf": rf,
        "gross_return": rf + gross_excess,
        "gross_excess_return": gross_excess,
        "net_return_primary": net_return,
        "net_excess_return_primary": gross_excess - primary_cost.total,
        "net_return_stress": stress_return,
        "net_excess_return_stress": gross_excess - stress_cost.total,
        "long_contribution": float(np.maximum(weights, 0.0) @ excess),
        "short_contribution": float(np.minimum(weights, 0.0) @ excess),
        "transaction_cost_primary": primary_cost.transaction,
        "borrow_cost_primary": primary_cost.borrow,
        "financing_cost_primary": primary_cost.financing,
        "total_cost_primary": primary_cost.total,
        "total_cost_stress": stress_cost.total,
        "forecast_volatility": solution.forecast_volatility,
        "long_gross": solution.long_gross,
        "short_gross": solution.short_gross,
        "gross": solution.gross,
        "net": solution.net,
        "turnover": solution.turnover,
        "active_long_holdings": int((weights > 1e-8).sum()),
        "active_short_holdings": int((weights < -1e-8).sum()),
        "eligible_long_candidates": int(long_mask.sum()),
        "eligible_short_candidates": int(short_mask.sum()),
        "maximum_weight": float(np.max(weights, initial=0.0)),
        "minimum_weight": float(np.min(weights, initial=0.0)),
        "maximum_adv_participation": float(np.max(participation, initial=0.0)),
        "latent_factors": latent_count,
        "latent_explained_residual_variance": latent_explained,
        "expected_return_scale": expected_scale,
        **{
            f"factor_exposure_{name}": float(solution.factor_exposures[index])
            for index, name in enumerate(FACTOR_NAMES)
        },
    }
    return positions, summary


def _checkpoint(output: Path, phase: str, profile: str, month: str) -> tuple[Path, Path]:
    root = output / "portfolios" / phase / profile / "months"
    return root / f"{month}.parquet", root / f"{month}.json"


def _run_profile(
    *,
    study: Mapping[str, Any],
    output: Path,
    phase: str,
    months: list[str],
    profile: Mapping[str, Any],
    identity: Mapping[str, str],
    inputs: base.PreparedInputs,
    loader: dict[str, Any],
    opportunity: Mapping[str, float],
    latent_history: pd.DataFrame,
    month_cache: dict[tuple[str, str], tuple[pd.DataFrame, np.ndarray]],
) -> pd.DataFrame:
    previous: dict[int, float] = {}
    nav = float(loader["nav_dollars"])
    summaries: list[dict[str, Any]] = []
    for month in months:
        parquet_path, receipt_path = _checkpoint(
            output, phase, str(profile["id"]), month
        )
        checkpoint_identity = {
            **identity,
            "phase": phase,
            "profile_identity": _canonical(profile),
        }
        if parquet_path.is_file() and receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            if any(receipt.get(key) != value for key, value in checkpoint_identity.items()):
                raise RuntimeError(f"V6.3 checkpoint mismatch: {receipt_path}")
            positions = pd.read_parquet(parquet_path)
            summary = dict(receipt["summary"])
            previous = {
                int(row.permno): float(row.end_drift_weight)
                for row in positions.itertuples(index=False)
                if abs(float(row.end_drift_weight)) > 1e-12
            }
            nav = float(summary["nav_end_primary"])
            summaries.append(summary)
            continue
        cache_key = (str(profile["signal_variant"]), month)
        if cache_key not in month_cache:
            month_cache[cache_key] = base._month_frame(inputs, loader, month)
        frame, covariance_root = month_cache[cache_key]
        positions, summary = _solve_profile_month(
            study=study,
            frame=frame.copy(),
            covariance_root=covariance_root,
            loader=loader,
            profile=profile,
            previous_map=previous,
            nav_dollars=nav,
            opportunity_percentile=float(opportunity.get(month, 0.50)),
            latent_history=latent_history,
        )
        _atomic_parquet(parquet_path, positions)
        _atomic_json(
            receipt_path,
            {
                "schema": "research_v6_3_portfolio_month_v1",
                **checkpoint_identity,
                "holding_month": month,
                "summary": summary,
            },
        )
        previous = {
            int(row.permno): float(row.end_drift_weight)
            for row in positions.itertuples(index=False)
            if abs(float(row.end_drift_weight)) > 1e-12
        }
        nav = float(summary["nav_end_primary"])
        summaries.append(summary)
        print(
            f"PORTFOLIO {phase} {profile['id']} {month} "
            f"net={summary['net_excess_return_primary']:.6f}",
            flush=True,
        )
    return pd.DataFrame(summaries).sort_values(
        "holding_month", kind="mergesort"
    ).reset_index(drop=True)


def _aggregate(frame: pd.DataFrame) -> dict[str, Any]:
    result = base._aggregate_variant(frame)
    net = frame["net_excess_return_primary"].to_numpy(float)
    stress = frame["net_excess_return_stress"].to_numpy(float)
    without_best_two = np.sort(net)[:-2] if len(net) > 2 else np.asarray([], dtype=float)
    result.update(
        {
            "profile": str(frame["profile"].iloc[0]),
            "idea": str(frame["idea"].iloc[0]),
            "annualized_long_contribution": float(frame["long_contribution"].mean() * 12.0),
            "annualized_short_contribution": float(frame["short_contribution"].mean() * 12.0),
            "annualized_net_excess_without_best_two_months": (
                float(without_best_two.mean() * 12.0)
                if len(without_best_two)
                else float("nan")
            ),
            "maximum_adv_participation": float(frame["maximum_adv_participation"].max()),
            "mean_expected_return_scale": float(frame["expected_return_scale"].mean()),
            "mean_latent_explained_residual_variance": float(
                frame["latent_explained_residual_variance"].mean()
            ),
            "positive_stress_month_fraction": float((stress > 0.0).mean()),
        }
    )
    return result


def _constraint_audit(
    output: Path,
    phase: str,
    profile: str,
    monthly: pd.DataFrame,
    loader: Mapping[str, Any],
) -> dict[str, Any]:
    optimizer = loader["optimizer"]
    maximum_candidate = 0.0
    maximum_reduction_increase = 0.0
    maximum_sic2_net = 0.0
    maximum_sic2_gross = 0.0
    root = output / "portfolios" / phase / profile / "months"
    for path in root.glob("*.parquet"):
        positions = pd.read_parquet(path)
        active = positions.loc[positions["weight"].ne(0.0)].copy()
        candidate = active["long_candidate"].astype(bool) | active["short_candidate"].astype(bool)
        if candidate.any():
            maximum_candidate = max(
                maximum_candidate,
                float(active.loc[candidate, "weight"].abs().max()),
            )
        inherited = ~candidate & active["previous_weight"].abs().gt(1e-12)
        if inherited.any():
            maximum_reduction_increase = max(
                maximum_reduction_increase,
                float(
                    (
                        active.loc[inherited, "weight"].abs()
                        - active.loc[inherited, "previous_weight"].abs()
                    ).max()
                ),
            )
        for _, group in active.groupby("sic2_industry_category", observed=True):
            weights = group["weight"].to_numpy(float)
            maximum_sic2_net = max(maximum_sic2_net, abs(float(weights.sum())))
            maximum_sic2_gross = max(maximum_sic2_gross, float(np.abs(weights).sum()))
    other = [f"factor_exposure_{name}" for name in FACTOR_NAMES[1:]]
    residuals = {
        "maximum_gross": float(monthly["gross"].max() - float(optimizer["maximum_gross"])),
        "minimum_long_gross": float(
            float(optimizer["minimum_long_gross"]) - monthly["long_gross"].min()
        ),
        "minimum_short_gross": float(
            float(optimizer["minimum_short_gross"]) - monthly["short_gross"].min()
        ),
        "maximum_net": float(monthly["net"].max() - float(optimizer["maximum_net"])),
        "minimum_net": float(float(optimizer["minimum_net"]) - monthly["net"].min()),
        "candidate_position": float(
            maximum_candidate - float(optimizer["maximum_position_weight"])
        ),
        "reduction_only_increase": float(maximum_reduction_increase),
        "turnover": float(
            monthly["turnover"].max() - float(optimizer["maximum_one_way_turnover"])
        ),
        "adv": float(
            monthly["maximum_adv_participation"].max()
            - float(optimizer["maximum_one_way_adv_participation"])
        ),
        "market_beta": float(
            monthly["factor_exposure_mktrf"].abs().max()
            - float(optimizer["market_beta_absolute_limit"])
        ),
        "other_factor": float(
            monthly[other].abs().to_numpy().max()
            - float(optimizer["other_factor_absolute_limit"])
        ),
        "sic2_net": float(
            maximum_sic2_net - float(optimizer["sic2_industry_net_absolute_limit"])
        ),
        "sic2_gross": float(
            maximum_sic2_gross - float(optimizer["sic2_industry_gross_limit"])
        ),
        "forecast_volatility": float(
            monthly["forecast_volatility"].max()
            - float(optimizer["primary_forecast_volatility"])
        ),
    }
    maximum = max(max(residuals.values()), 0.0)
    return {
        "breach_residuals": residuals,
        "maximum_positive_breach": maximum,
        "passes_practical_tolerance": maximum <= 5e-7,
        "practical_tolerance": 5e-7,
    }


def _select_idea_winners(
    validation: pd.DataFrame,
    profiles: list[dict[str, Any]],
    study: Mapping[str, Any],
) -> dict[str, str]:
    control = validation.loc[validation["profile"].eq("incumbent_1p5")].iloc[0]
    minimum_sharpe = (
        float(control["net_sharpe_primary"])
        * float(study["selection"]["guardrails"]["minimum_net_sharpe_fraction_of_incumbent"])
    )
    order = {profile["id"]: index for index, profile in enumerate(profiles)}
    winners: dict[str, str] = {}
    for idea in [f"0{index}_" for index in range(1, 7)]:
        idea_name = next(
            profile["idea"]
            for profile in profiles
            if str(profile["idea"]).startswith(idea)
        )
        candidates = validation.loc[validation["idea"].eq(idea_name)].copy()
        candidates = candidates.loc[
            candidates["annualized_net_excess_return_stress"].gt(0.0)
            & candidates["net_sharpe_primary"].ge(minimum_sharpe)
            & candidates["constraints_pass"].astype(bool)
        ]
        fallback = validation.loc[validation["profile"].eq("incumbent_1p5")].copy()
        candidates = pd.concat([fallback, candidates], ignore_index=True)
        candidates["profile_order"] = candidates["profile"].map(order).fillna(-1)
        winner = candidates.sort_values(
            ["annualized_net_excess_return_primary", "profile_order"],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[0]
        winners[idea_name] = str(winner["profile"])
    return winners


def _write_result_markdown(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# Research V6.3 — six signal-to-weight experiments",
        "",
        "## Decision",
        "",
        str(result["decision"]),
        "",
        "The frozen V6.1 1.5% portfolio is the common comparator. All results are",
        "post-diagnostic historical development, not untouched confirmation.",
        "",
        "## Historical-test comparison",
        "",
        "| Idea | Selected profile | Annual net excess | Net Sharpe | Stress Sharpe | Max drawdown |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for idea, profile in result["idea_winners"].items():
        summary = result["historical_test"].get(profile)
        if summary is None:
            continue
        lines.append(
            f"| {idea} | {profile} | {summary['annualized_net_excess_return_primary']:.2%} "
            f"| {summary['net_sharpe_primary']:.3f} | {summary['net_sharpe_stress']:.3f} "
            f"| {summary['maximum_drawdown_primary']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Research boundary",
            "",
            "Validation is February 2022–December 2023. The historical test is",
            "January 2024–August 2025. September 2025–August 2026 remains quarantined;",
            "clean confirmation cannot begin before September 2026.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def run(
    config_path: Path,
    output_root: Path,
    *,
    smoke: bool = False,
    signals_only: bool = False,
) -> Path:
    study, frozen, paths, input_identity = _verify(config_path)
    code_identity = _code_identity()
    config_sha = _sha256(config_path)
    output = output_root / f"{config_sha[:12]}-{input_identity[:12]}-{code_identity[:12]}"
    if smoke:
        output = output / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "config_sha256": config_sha,
        "input_identity": input_identity,
        "code_identity": code_identity,
    }
    _atomic_json(
        output / "run_manifest.json",
        {
            "schema": "research_v6_3_signal_weight_run_v1",
            "status": "RUNNING",
            **identity,
            "ideas": list(study["ideas"]),
        },
    )
    signal_paths = _prepare_signals(
        study, paths, output, _canonical(identity), smoke=smoke
    )
    if signals_only:
        _atomic_json(
            output / "run_manifest.json",
            {
                "schema": "research_v6_3_signal_weight_run_v1",
                "status": "SIGNALS_COMPLETE",
                **identity,
                "ideas": list(study["ideas"]),
            },
        )
        print(f"SIGNALS COMPLETE {output}", flush=True)
        return output

    variants, loader = _input_variants(study, paths, signal_paths, frozen)
    all_profiles = _profiles(study)
    profiles = all_profiles
    validation_months = pd.period_range(
        study["evidence"]["validation_start"],
        study["evidence"]["validation_end"],
        freq="M",
    ).astype(str).tolist()
    test_months = pd.period_range(
        study["evidence"]["historical_test_start"],
        study["evidence"]["historical_test_end"],
        freq="M",
    ).astype(str).tolist()
    if smoke:
        validation_months = validation_months[:1]
        test_months = test_months[:1]
        wanted_ideas = {
            "control",
            "01_confidence_adjusted_edge",
            "02_opportunity_breadth_scaling",
            "06_latent_risk_layer",
        }
        profiles = []
        seen: set[str] = set()
        for profile in all_profiles:
            if profile["idea"] in wanted_ideas and profile["idea"] not in seen:
                profiles.append(profile)
                seen.add(str(profile["idea"]))

    opportunity = _opportunity_map(variants["incumbent"])
    latent_history = _latent_history(variants["incumbent"])
    month_cache: dict[tuple[str, str], tuple[pd.DataFrame, np.ndarray]] = {}
    validation_frames: dict[str, pd.DataFrame] = {}
    validation_summaries: list[dict[str, Any]] = []
    validation_audits: dict[str, dict[str, Any]] = {}
    validation_failures: dict[str, str] = {}
    for profile in profiles:
        profile_id = str(profile["id"])
        inputs = variants[str(profile["signal_variant"])]
        try:
            frame = _run_profile(
                study=study,
                output=output,
                phase="validation",
                months=validation_months,
                profile=profile,
                identity=identity,
                inputs=inputs,
                loader=loader,
                opportunity=opportunity,
                latent_history=latent_history,
                month_cache=month_cache,
            )
        except RuntimeError as error:
            # An alternative signal can make the frozen long/short/risk
            # contract infeasible (for example, by admitting no gated longs).
            # That is a valid failure of the challenger, not authority to
            # relax the incumbent constraints after observing validation.
            if "infeasible" not in str(error).lower():
                raise
            validation_failures[profile_id] = str(error)
            validation_summaries.append(
                {
                    "profile": profile_id,
                    "idea": str(profile["idea"]),
                    "annualized_net_excess_return_primary": float("nan"),
                    "annualized_net_excess_return_stress": float("nan"),
                    "net_sharpe_primary": float("nan"),
                    "net_sharpe_stress": float("nan"),
                    "constraints_pass": False,
                    "maximum_constraint_breach": float("inf"),
                    "validation_status": "INFEASIBLE_FROZEN_PORTFOLIO_CONTRACT",
                }
            )
            print(f"VALIDATION {profile_id} INFEASIBLE: {error}", flush=True)
            continue
        validation_frames[profile_id] = frame
        summary = _aggregate(frame)
        audit = _constraint_audit(
            output, "validation", profile_id, frame, loader
        )
        summary["constraints_pass"] = bool(audit["passes_practical_tolerance"])
        summary["maximum_constraint_breach"] = float(audit["maximum_positive_breach"])
        validation_summaries.append(summary)
        validation_audits[profile_id] = audit
        print(
            f"VALIDATION {profile_id} return="
            f"{summary['annualized_net_excess_return_primary']:.6f} "
            f"sharpe={summary['net_sharpe_primary']:.6f}",
            flush=True,
        )
    validation_monthly = pd.concat(validation_frames.values(), ignore_index=True)
    validation_summary = pd.DataFrame(validation_summaries)
    _atomic_parquet(output / "validation_monthly.parquet", validation_monthly)
    _atomic_parquet(output / "validation_summary.parquet", validation_summary)
    _atomic_json(output / "validation_constraint_audits.json", validation_audits)
    _atomic_json(output / "validation_failures.json", validation_failures)
    if smoke:
        _atomic_json(
            output / "run_manifest.json",
            {
                "schema": "research_v6_3_signal_weight_run_v1",
                "status": "SMOKE_COMPLETE",
                **identity,
                "validation_profiles": [profile["id"] for profile in profiles],
            },
        )
        print(f"SMOKE COMPLETE {output}", flush=True)
        return output

    winners = _select_idea_winners(validation_summary, profiles, study)
    _atomic_json(output / "idea_winners.json", winners)
    profile_lookup = {str(profile["id"]): profile for profile in profiles}
    selected_ids = list(dict.fromkeys(["incumbent_1p5", *winners.values()]))
    test_frames: dict[str, pd.DataFrame] = {}
    test_summaries: dict[str, dict[str, Any]] = {}
    test_audits: dict[str, dict[str, Any]] = {}
    for profile_id in selected_ids:
        profile = profile_lookup[profile_id]
        frame = _run_profile(
            study=study,
            output=output,
            phase="historical_test",
            months=test_months,
            profile=profile,
            identity=identity,
            inputs=variants[str(profile["signal_variant"])],
            loader=loader,
            opportunity=opportunity,
            latent_history=latent_history,
            month_cache=month_cache,
        )
        test_frames[profile_id] = frame
        summary = _aggregate(frame)
        audit = _constraint_audit(
            output, "historical_test", profile_id, frame, loader
        )
        summary["constraints_pass"] = bool(audit["passes_practical_tolerance"])
        summary["maximum_constraint_breach"] = float(audit["maximum_positive_breach"])
        test_summaries[profile_id] = summary
        test_audits[profile_id] = audit
        print(
            f"TEST {profile_id} return="
            f"{summary['annualized_net_excess_return_primary']:.6f} "
            f"sharpe={summary['net_sharpe_primary']:.6f}",
            flush=True,
        )
    test_monthly = pd.concat(test_frames.values(), ignore_index=True)
    _atomic_parquet(output / "historical_test_monthly.parquet", test_monthly)
    _atomic_parquet(
        output / "historical_test_summary.parquet",
        pd.DataFrame(test_summaries.values()),
    )
    _atomic_json(output / "historical_test_constraint_audits.json", test_audits)

    control = test_frames["incumbent_1p5"]
    paired = {
        profile_id: inference_tools._paired_inference(
            frame, control, study["inference"]
        )
        for profile_id, frame in test_frames.items()
    }
    incumbent_summary = test_summaries["incumbent_1p5"]
    promising: list[str] = []
    for profile_id, summary in test_summaries.items():
        if profile_id == "incumbent_1p5":
            continue
        if (
            summary["annualized_net_excess_return_primary"]
            > incumbent_summary["annualized_net_excess_return_primary"]
            and summary["net_sharpe_primary"] > incumbent_summary["net_sharpe_primary"]
            and summary["net_sharpe_stress"] > 0.0
            and summary["constraints_pass"]
            and paired[profile_id]["annualized_mean_net_return_difference"] > 0.0
        ):
            promising.append(profile_id)
    decision = (
        "PROMISING_CHALLENGER_REQUIRES_UNTOUCHED_FORWARD_CONFIRMATION"
        if promising
        else "RETAIN_V6_1_1P5_INCUMBENT"
    )

    frontier = pd.read_parquet(
        ROOT / study["incumbent"]["frontier_root"] / "historical_test_monthly.parquet"
    )
    frontier = frontier.loc[
        frontier["profile"].astype(str).eq(study["incumbent"]["frontier_profile"])
    ].sort_values("holding_month", kind="mergesort")
    reproduced = control.sort_values("holding_month", kind="mergesort")
    joined = reproduced[["holding_month", "net_excess_return_primary"]].merge(
        frontier[["holding_month", "net_excess_return_primary"]],
        on="holding_month",
        suffixes=("_v6_3", "_frontier"),
        validate="one_to_one",
    )
    control_reproduction_error = float(
        np.max(
            np.abs(
                joined["net_excess_return_primary_v6_3"].to_numpy(float)
                - joined["net_excess_return_primary_frontier"].to_numpy(float)
            ),
            initial=0.0,
        )
    )
    if control_reproduction_error > 1e-10:
        raise RuntimeError(
            f"V6.3 incumbent failed exact monthly reproduction: {control_reproduction_error}"
        )
    result = {
        "schema": "research_v6_3_signal_weight_result_v1",
        "status": "COMPLETE",
        "decision": decision,
        "promising_challengers": promising,
        "idea_winners": winners,
        "validation": {row["profile"]: row for row in validation_summaries},
        "historical_test": test_summaries,
        "paired_inference_vs_incumbent": paired,
        "validation_failures": validation_failures,
        "control_reproduction_max_absolute_monthly_error": control_reproduction_error,
        "all_six_ideas_executed": True,
        "historical_test_never_selected_within_idea_profile": True,
        "post_diagnostic_historical_development_only": True,
        "quarantined_start": study["evidence"]["quarantined_start"],
        "untouched_confirmation_earliest_start": study["evidence"][
            "untouched_confirmation_earliest_start"
        ],
    }
    _atomic_json(output / "result.json", result)
    _write_result_markdown(output / "RESULT.md", result)
    _atomic_json(
        output / "run_manifest.json",
        {
            "schema": "research_v6_3_signal_weight_run_v1",
            "status": "COMPLETE",
            **identity,
            "validation_profile_months": len(profiles) * len(validation_months),
            "historical_test_profiles": selected_ids,
            "historical_test_profile_months": len(selected_ids) * len(test_months),
            "idea_winners": winners,
        },
    )
    print(json.dumps(result, indent=2, default=str), flush=True)
    print(f"COMPLETE {output}", flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--signals-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run(
            args.config.resolve(),
            args.output_root.resolve(),
            smoke=bool(args.smoke),
            signals_only=bool(args.signals_only),
        )
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
