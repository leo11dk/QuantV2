from __future__ import annotations

import numpy as np
import pandas as pd


VALID_PREDICTIONS = {"up", "down", "no_trade"}

PREDICTION_METRIC_COLUMNS = [
    "horizon",
    "total_count",
    "evaluated_count",
    "missing_label_count",
    "trade_count",
    "no_trade_count",
    "coverage",
    "hit_count",
    "hit_rate",
    "mean_signed_forward_return",
    "median_signed_forward_return",
    "std_signed_forward_return",
    "mean_forward_return_all",
    "mean_forward_return_traded",
]


def evaluate_baseline_predictions(
    data: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 5),
    group_cols: tuple[str, ...] | None = None,
    prediction_col: str = "predicted_direction",
    trade_allowed_col: str = "trade_allowed",
    no_trade_reason_col: str = "no_trade_reason",
) -> pd.DataFrame:
    """Evaluate deterministic baseline predictions against existing labels.

    The evaluator only aggregates existing ``forward_return_Nd`` label columns.
    It does not create labels, inspect price columns, use label-date columns,
    train models, create orders, or simulate execution.
    """
    label_columns = [f"forward_return_{horizon}d" for horizon in horizons]
    _validate_required_columns(
        data,
        columns=(prediction_col, trade_allowed_col, no_trade_reason_col),
        label="required prediction columns",
    )
    _validate_required_columns(
        data,
        columns=tuple(label_columns),
        label="required forward-return columns",
    )
    _validate_group_columns(data, group_cols)
    _validate_prediction_values(data[prediction_col], prediction_col)

    trade_allowed = _coerce_boolean_like(data[trade_allowed_col], trade_allowed_col)
    _validate_prediction_consistency(
        predictions=data[prediction_col],
        trade_allowed=trade_allowed,
        prediction_col=prediction_col,
        trade_allowed_col=trade_allowed_col,
    )

    rows: list[dict[str, object]] = []
    helper_col = _unused_helper_column(data)
    working = data.copy(deep=False)
    working[helper_col] = trade_allowed.to_numpy()

    for horizon, label_column in zip(horizons, label_columns):
        if group_cols is None:
            rows.append(
                _prediction_metric_row(
                    predictions=working[prediction_col],
                    trade_allowed=working[helper_col],
                    forward_returns=working[label_column],
                    horizon=horizon,
                )
            )
            continue

        groupby_arg: str | list[str]
        groupby_arg = group_cols[0] if len(group_cols) == 1 else list(group_cols)
        grouped = working.groupby(groupby_arg, sort=False, dropna=False)

        for group_key, group in grouped:
            key_values = (group_key,) if len(group_cols) == 1 else tuple(group_key)
            row = {
                group_col: key_value
                for group_col, key_value in zip(group_cols, key_values)
            }
            row.update(
                _prediction_metric_row(
                    predictions=group[prediction_col],
                    trade_allowed=group[helper_col],
                    forward_returns=group[label_column],
                    horizon=horizon,
                )
            )
            rows.append(row)

    columns = (
        PREDICTION_METRIC_COLUMNS
        if group_cols is None
        else [*group_cols, *PREDICTION_METRIC_COLUMNS]
    )
    return pd.DataFrame(rows, columns=columns)


def _validate_required_columns(
    data: pd.DataFrame,
    columns: tuple[str, ...],
    label: str,
) -> None:
    missing_columns = [column for column in columns if column not in data.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"data is missing {label}: {missing}")


def _validate_group_columns(
    data: pd.DataFrame,
    group_cols: tuple[str, ...] | None,
) -> None:
    if group_cols is None:
        return
    if not group_cols:
        raise ValueError("group_cols must contain at least one column")

    missing_columns = [column for column in group_cols if column not in data.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"data is missing group columns: {missing}")

    label_date_columns = [column for column in group_cols if "label_date" in column]
    if label_date_columns:
        columns = ", ".join(label_date_columns)
        raise ValueError(f"group_cols cannot include label_date columns: {columns}")


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


def _prediction_metric_row(
    predictions: pd.Series,
    trade_allowed: pd.Series,
    forward_returns: pd.Series,
    horizon: int,
) -> dict[str, object]:
    valid_label = forward_returns.notna()
    traded = trade_allowed & valid_label
    no_trade = ~trade_allowed & valid_label

    total_count = int(len(forward_returns))
    evaluated_count = int(valid_label.sum())
    missing_label_count = int(forward_returns.isna().sum())
    trade_count = int(traded.sum())
    no_trade_count = int(no_trade.sum())

    traded_returns = forward_returns[traded]
    traded_predictions = predictions[traded]
    hit_mask = (
        traded_predictions.eq("up")
        & traded_returns.gt(0.0)
        | traded_predictions.eq("down")
        & traded_returns.lt(0.0)
    )
    hit_count = int(hit_mask.sum())

    signed_forward_returns = traded_returns.where(
        traded_predictions.eq("up"),
        -traded_returns,
    )

    return {
        "horizon": int(horizon),
        "total_count": total_count,
        "evaluated_count": evaluated_count,
        "missing_label_count": missing_label_count,
        "trade_count": trade_count,
        "no_trade_count": no_trade_count,
        "coverage": trade_count / evaluated_count if evaluated_count else np.nan,
        "hit_count": hit_count,
        "hit_rate": hit_count / trade_count if trade_count else np.nan,
        "mean_signed_forward_return": signed_forward_returns.mean(),
        "median_signed_forward_return": signed_forward_returns.median(),
        "std_signed_forward_return": signed_forward_returns.std(),
        "mean_forward_return_all": forward_returns[valid_label].mean(),
        "mean_forward_return_traded": traded_returns.mean(),
    }


def _format_values(values: pd.Series) -> str:
    formatted = sorted({repr(value) for value in values.tolist()})
    return ", ".join(formatted)


def _unused_helper_column(data: pd.DataFrame) -> str:
    column = "__quantv2_trade_allowed_bool"
    while column in data.columns:
        column = f"{column}_"
    return column
