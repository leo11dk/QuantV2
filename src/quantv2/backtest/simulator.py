from __future__ import annotations

import pandas as pd

from quantv2.backtest.costs import add_cost_adjusted_returns
from quantv2.backtest.walk_forward import WalkForwardSplit, make_walk_forward_splits
from quantv2.evaluation.prediction_metrics import evaluate_baseline_predictions
from quantv2.models.rule_baseline import generate_rule_baseline_predictions


SPLIT_COLUMNS = [
    "split_id",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "train_row_count",
    "test_row_count",
]

COST_ADJUSTED_SUMMARY_COLUMNS = [
    "split_id",
    "horizon",
    "trade_count",
    "missing_cost_adjusted_count",
    "mean_signed_forward_return",
    "mean_cost_adjusted_signed_forward_return",
    "median_cost_adjusted_signed_forward_return",
    "std_cost_adjusted_signed_forward_return",
    "min_cost_adjusted_signed_forward_return",
    "max_cost_adjusted_signed_forward_return",
]

FORBIDDEN_OUTPUT_COLUMNS = {
    "order",
    "execution",
    "fill",
    "position",
    "pnl",
    "profit",
    "brokerage",
}


def run_walk_forward_baseline_evaluation(
    data: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 5),
    train_window: int = 252,
    test_window: int = 21,
    step_size: int | None = None,
    min_train_size: int | None = None,
    date_col: str = "decision_date",
    prediction_kwargs: dict | None = None,
    cost_kwargs: dict | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the deterministic baseline over walk-forward test windows.

    The current rule baseline does not train, so training rows are used only
    for point-in-time split metadata. Predictions, metrics, and cost-adjusted
    research return estimates are produced from each split's test rows only.
    """

    resolved_prediction_kwargs = (
        {} if prediction_kwargs is None else dict(prediction_kwargs)
    )
    resolved_prediction_kwargs = {
        "decision_date_col": date_col,
        **resolved_prediction_kwargs,
    }
    resolved_cost_kwargs = {} if cost_kwargs is None else dict(cost_kwargs)

    splits = make_walk_forward_splits(
        data=data,
        date_col=date_col,
        train_window=train_window,
        test_window=test_window,
        step_size=step_size,
        min_train_size=min_train_size,
    )

    split_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for split in splits:
        split_rows.append(_split_row(split))

        test_rows = data.loc[split.test_indices].copy(deep=True)
        split_predictions = generate_rule_baseline_predictions(
            test_rows,
            **resolved_prediction_kwargs,
        )
        split_predictions = add_cost_adjusted_returns(
            split_predictions,
            horizons=horizons,
            **resolved_cost_kwargs,
        )
        split_predictions = _add_split_metadata(split_predictions, split)
        prediction_frames.append(split_predictions)

    splits_frame = pd.DataFrame(split_rows, columns=SPLIT_COLUMNS)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    prediction_summary = evaluate_baseline_predictions(
        predictions,
        horizons=horizons,
        group_cols=("split_id",),
    )
    cost_adjusted_summary = _summarize_cost_adjusted_returns(
        predictions,
        horizons=horizons,
    )

    outputs = {
        "splits": splits_frame,
        "predictions": predictions,
        "prediction_summary": prediction_summary,
        "cost_adjusted_summary": cost_adjusted_summary,
    }
    _validate_no_forbidden_output_columns(outputs)

    return outputs


def _split_row(split: WalkForwardSplit) -> dict[str, object]:
    return {
        "split_id": split.split_id,
        "train_start": split.train_start,
        "train_end": split.train_end,
        "test_start": split.test_start,
        "test_end": split.test_end,
        "train_row_count": len(split.train_indices),
        "test_row_count": len(split.test_indices),
    }


def _add_split_metadata(
    predictions: pd.DataFrame,
    split: WalkForwardSplit,
) -> pd.DataFrame:
    result = predictions.copy(deep=True)
    result["split_id"] = split.split_id
    result["train_start"] = split.train_start
    result["train_end"] = split.train_end
    result["test_start"] = split.test_start
    result["test_end"] = split.test_end
    return result


def _summarize_cost_adjusted_returns(
    predictions: pd.DataFrame,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for split_id, split_predictions in predictions.groupby(
        "split_id",
        sort=False,
        dropna=False,
    ):
        for horizon in horizons:
            signed_column = f"signed_forward_return_{horizon}d"
            adjusted_column = f"cost_adjusted_signed_forward_return_{horizon}d"
            signed_returns = pd.to_numeric(
                split_predictions[signed_column],
                errors="coerce",
            )
            adjusted_returns = pd.to_numeric(
                split_predictions[adjusted_column],
                errors="coerce",
            )

            rows.append(
                {
                    "split_id": split_id,
                    "horizon": int(horizon),
                    "trade_count": int(adjusted_returns.notna().sum()),
                    "missing_cost_adjusted_count": int(adjusted_returns.isna().sum()),
                    "mean_signed_forward_return": signed_returns.mean(),
                    "mean_cost_adjusted_signed_forward_return": adjusted_returns.mean(),
                    "median_cost_adjusted_signed_forward_return": (
                        adjusted_returns.median()
                    ),
                    "std_cost_adjusted_signed_forward_return": adjusted_returns.std(),
                    "min_cost_adjusted_signed_forward_return": adjusted_returns.min(),
                    "max_cost_adjusted_signed_forward_return": adjusted_returns.max(),
                }
            )

    return pd.DataFrame(rows, columns=COST_ADJUSTED_SUMMARY_COLUMNS)


def _validate_no_forbidden_output_columns(
    outputs: dict[str, pd.DataFrame],
) -> None:
    for name, frame in outputs.items():
        forbidden_columns = sorted(
            FORBIDDEN_OUTPUT_COLUMNS.intersection(frame.columns)
        )
        if forbidden_columns:
            forbidden = ", ".join(forbidden_columns)
            raise ValueError(f"{name} contains forbidden output columns: {forbidden}")
