from __future__ import annotations

import pandas as pd


POSITIVE_EVENT_DIRECTIONS = {
    "positive",
    "bullish",
    "up",
    "beat",
    "upgrade",
}
NEGATIVE_EVENT_DIRECTIONS = {
    "negative",
    "bearish",
    "down",
    "miss",
    "downgrade",
}


def generate_rule_baseline_predictions(
    data: pd.DataFrame,
    gap_threshold: float = 0.01,
    momentum_threshold: float = 0.02,
    min_score_to_trade: int = 2,
    max_volatility: float | None = None,
    ticker_col: str = "ticker",
    decision_date_col: str = "decision_date",
    gap_col: str = "gap_pct",
    prior_return_col: str = "prior_5d_return",
    volatility_col: str = "volatility_20d",
    event_direction_col: str = "event_direction",
) -> pd.DataFrame:
    """Create deterministic rule-based baseline predictions.

    The baseline scores only point-in-time feature columns already present in
    ``data``. It does not train a model, create labels, infer missing features,
    or inspect forward-return and future-close label columns.
    """
    required_columns = {
        ticker_col,
        decision_date_col,
        gap_col,
        prior_return_col,
    }
    if max_volatility is not None:
        required_columns.add(volatility_col)

    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"data is missing required columns: {missing}")

    predictions = data.copy(deep=True)
    predictions = predictions.sort_values(
        [ticker_col, decision_date_col],
        kind="mergesort",
    )
    predictions = predictions.reset_index(drop=True)

    gap = pd.to_numeric(predictions[gap_col], errors="coerce")
    prior_return = pd.to_numeric(predictions[prior_return_col], errors="coerce")
    missing_features = gap.isna() | prior_return.isna()

    high_volatility = pd.Series(False, index=predictions.index)
    if max_volatility is not None:
        volatility = pd.to_numeric(predictions[volatility_col], errors="coerce")
        missing_features = missing_features | volatility.isna()
        high_volatility = volatility.gt(max_volatility)

    scores = pd.Series(0.0, index=predictions.index)
    scores = scores + gap.gt(gap_threshold).astype(int)
    scores = scores - gap.lt(-gap_threshold).astype(int)
    scores = scores + prior_return.gt(momentum_threshold).astype(int)
    scores = scores - prior_return.lt(-momentum_threshold).astype(int)

    if event_direction_col in predictions.columns:
        scores = scores + predictions[event_direction_col].map(_event_direction_score)

    scores = scores.mask(missing_features)

    predicted_direction = pd.Series("no_trade", index=predictions.index, dtype="object")
    predicted_direction.loc[scores.gt(0)] = "up"
    predicted_direction.loc[scores.lt(0)] = "down"

    trade_allowed = (
        scores.abs().ge(min_score_to_trade)
        & ~missing_features
        & ~high_volatility
    )

    no_trade_reason = pd.Series("weak_signal", index=predictions.index, dtype="object")
    no_trade_reason.loc[trade_allowed] = "trade_allowed"
    no_trade_reason.loc[high_volatility & ~missing_features] = "high_volatility"
    no_trade_reason.loc[missing_features] = "missing_features"

    predictions["rule_score"] = scores
    predictions["predicted_direction"] = predicted_direction
    predictions["trade_allowed"] = trade_allowed
    predictions["no_trade_reason"] = no_trade_reason

    return predictions


def _event_direction_score(value: object) -> int:
    if pd.isna(value):
        return 0

    normalized = str(value).strip().lower()
    if normalized in POSITIVE_EVENT_DIRECTIONS:
        return 1
    if normalized in NEGATIVE_EVENT_DIRECTIONS:
        return -1

    numeric_value = pd.to_numeric(normalized, errors="coerce")
    if pd.isna(numeric_value):
        return 0
    if numeric_value > 0:
        return 1
    if numeric_value < 0:
        return -1
    return 0
