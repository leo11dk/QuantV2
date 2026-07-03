from __future__ import annotations

import numpy as np
import pandas as pd


VALID_PREDICTIONS = {"up", "down", "no_trade"}


def calculate_round_trip_cost_rate(
    commission_bps: float = 0.0,
    slippage_bps: float = 5.0,
    extra_cost_bps: float = 0.0,
) -> float:
    """Convert simple round-trip cost assumptions from bps to a return rate.

    The estimate is a research assumption only. It does not model execution,
    borrow fees, market impact, latency, partial fills, or real profitability.
    """
    _validate_non_negative_bps(
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        extra_cost_bps=extra_cost_bps,
    )
    return (2 * commission_bps + 2 * slippage_bps + extra_cost_bps) / 10_000


def add_cost_adjusted_returns(
    data: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 5),
    prediction_col: str = "predicted_direction",
    trade_allowed_col: str = "trade_allowed",
    commission_bps: float = 0.0,
    slippage_bps: float = 5.0,
    extra_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Add cost-adjusted directional return estimates for evaluation.

    This function only adjusts existing ``forward_return_Nd`` label columns.
    It does not create labels from prices, inspect label-date columns, train
    models, create live trades, or simulate orders/execution/PnL.
    """
    label_columns = tuple(f"forward_return_{horizon}d" for horizon in horizons)
    _validate_required_columns(
        data,
        columns=(prediction_col, trade_allowed_col),
        label="required prediction columns",
    )
    _validate_required_columns(
        data,
        columns=label_columns,
        label="required forward-return columns",
    )
    _validate_prediction_values(data[prediction_col], prediction_col)

    trade_allowed = _coerce_boolean_like(data[trade_allowed_col], trade_allowed_col)
    _validate_prediction_consistency(
        predictions=data[prediction_col],
        trade_allowed=trade_allowed,
        prediction_col=prediction_col,
        trade_allowed_col=trade_allowed_col,
    )

    round_trip_cost_rate = calculate_round_trip_cost_rate(
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        extra_cost_bps=extra_cost_bps,
    )

    result = data.copy(deep=True)
    result["estimated_round_trip_cost_rate"] = round_trip_cost_rate

    for horizon, label_column in zip(horizons, label_columns):
        signed_column = f"signed_forward_return_{horizon}d"
        adjusted_column = f"cost_adjusted_signed_forward_return_{horizon}d"

        forward_returns = data[label_column]
        traded_with_label = trade_allowed & forward_returns.notna()
        signed_returns = pd.Series(np.nan, index=data.index, dtype="float64")

        up_mask = traded_with_label & data[prediction_col].eq("up")
        down_mask = traded_with_label & data[prediction_col].eq("down")
        signed_returns.loc[up_mask] = forward_returns.loc[up_mask].astype("float64")
        signed_returns.loc[down_mask] = -forward_returns.loc[down_mask].astype(
            "float64"
        )

        result[signed_column] = signed_returns
        result[adjusted_column] = signed_returns - round_trip_cost_rate

    return result


def _validate_non_negative_bps(**bps_values: float) -> None:
    for name, value in bps_values.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")


def _validate_required_columns(
    data: pd.DataFrame,
    columns: tuple[str, ...],
    label: str,
) -> None:
    missing_columns = [column for column in columns if column not in data.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"data is missing {label}: {missing}")


def _validate_prediction_values(
    predictions: pd.Series,
    prediction_col: str,
) -> None:
    invalid_mask = ~predictions.isin(VALID_PREDICTIONS)
    if invalid_mask.any():
        invalid = _format_values(predictions[invalid_mask])
        expected = ", ".join(sorted(VALID_PREDICTIONS))
        raise ValueError(
            f"{prediction_col} contains invalid values: {invalid}; "
            f"expected one of {expected}"
        )


def _coerce_boolean_like(values: pd.Series, column: str) -> pd.Series:
    coerced_values: list[bool] = []
    invalid_values: list[object] = []

    for value in values:
        coerced = _coerce_bool(value)
        if coerced is None:
            invalid_values.append(value)
            coerced_values.append(False)
            continue
        coerced_values.append(coerced)

    if invalid_values:
        invalid = _format_values(pd.Series(invalid_values, dtype="object"))
        raise ValueError(f"{column} must be boolean-like; invalid values: {invalid}")

    return pd.Series(coerced_values, index=values.index, dtype="bool")


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)

    if isinstance(value, (float, np.floating)):
        if pd.isna(value) or value not in (0.0, 1.0):
            return None
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "1", "yes", "y"}:
            return True
        if normalized in {"false", "f", "0", "no", "n"}:
            return False

    return None


def _validate_prediction_consistency(
    predictions: pd.Series,
    trade_allowed: pd.Series,
    prediction_col: str,
    trade_allowed_col: str,
) -> None:
    invalid_mask = trade_allowed & predictions.eq("no_trade")
    if invalid_mask.any():
        raise ValueError(
            f"{trade_allowed_col}=True requires {prediction_col} to be up or down"
        )


def _format_values(values: pd.Series) -> str:
    formatted = sorted({repr(value) for value in values.tolist()})
    return ", ".join(formatted)
