import pandas as pd
import pytest

from quantv2.backtest.simulator import run_walk_forward_baseline_evaluation


HORIZONS = (1, 3, 5)
FORBIDDEN_COLUMNS = {
    "order",
    "execution",
    "fill",
    "position",
    "pnl",
    "profit",
    "brokerage",
}


def _sample_data(
    num_dates: int = 8,
    tickers: tuple[str, ...] = ("AAA", "BBB"),
) -> pd.DataFrame:
    rows = []
    row_id = 0

    for date_idx, decision_date in enumerate(
        pd.date_range("2024-01-01", periods=num_dates, freq="D")
    ):
        for ticker_idx, ticker in enumerate(tickers):
            signal_bucket = (date_idx + ticker_idx) % 3
            if signal_bucket == 0:
                gap_pct = 0.02
                prior_5d_return = 0.03
                event_direction = "positive"
            elif signal_bucket == 1:
                gap_pct = -0.02
                prior_5d_return = -0.03
                event_direction = "negative"
            else:
                gap_pct = 0.0
                prior_5d_return = 0.0
                event_direction = "neutral"

            return_sign = 1 if (date_idx + ticker_idx) % 2 == 0 else -1
            rows.append(
                {
                    "row_id": row_id,
                    "ticker": ticker,
                    "decision_date": decision_date,
                    "gap_pct": gap_pct,
                    "prior_5d_return": prior_5d_return,
                    "volatility_20d": 0.10,
                    "event_direction": event_direction,
                    "forward_return_1d": 0.01 * return_sign,
                    "forward_return_3d": 0.015 * return_sign,
                    "forward_return_5d": 0.02 * return_sign,
                    "label_date_1d": decision_date + pd.Timedelta(days=1),
                    "label_date_3d": decision_date + pd.Timedelta(days=3),
                    "label_date_5d": decision_date + pd.Timedelta(days=5),
                }
            )
            row_id += 1

    return pd.DataFrame(rows)


def _run(data: pd.DataFrame, **kwargs: object) -> dict[str, pd.DataFrame]:
    return run_walk_forward_baseline_evaluation(
        data,
        train_window=3,
        test_window=2,
        step_size=2,
        **kwargs,
    )


def _split_horizon_pairs(frame: pd.DataFrame) -> set[tuple[int, int]]:
    return {
        (int(row["split_id"]), int(row["horizon"]))
        for _, row in frame[["split_id", "horizon"]].iterrows()
    }


def test_runner_returns_expected_frames_and_test_window_predictions() -> None:
    data = _sample_data()

    result = _run(data)

    assert set(result) == {
        "splits",
        "predictions",
        "prediction_summary",
        "cost_adjusted_summary",
    }

    splits = result["splits"]
    predictions = result["predictions"]

    assert splits["split_id"].tolist() == [0, 1]
    assert len(splits) == 2
    assert splits["train_row_count"].tolist() == [6, 6]
    assert splits["test_row_count"].tolist() == [4, 4]
    assert splits["train_end"].lt(splits["test_start"]).all()

    assert len(predictions) == 8
    assert set(data.columns).issubset(predictions.columns)

    split_metadata_columns = {
        "split_id",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
    }
    rule_columns = {
        "rule_score",
        "predicted_direction",
        "trade_allowed",
        "no_trade_reason",
    }
    cost_columns = {"estimated_round_trip_cost_rate"}
    for horizon in HORIZONS:
        cost_columns.add(f"signed_forward_return_{horizon}d")
        cost_columns.add(f"cost_adjusted_signed_forward_return_{horizon}d")

    assert split_metadata_columns.issubset(predictions.columns)
    assert rule_columns.issubset(predictions.columns)
    assert cost_columns.issubset(predictions.columns)

    for split in splits.itertuples(index=False):
        split_predictions = predictions[predictions["split_id"].eq(split.split_id)]
        decision_dates = pd.to_datetime(split_predictions["decision_date"])

        assert len(split_predictions) == split.test_row_count
        assert decision_dates.min() >= split.test_start
        assert decision_dates.max() <= split.test_end
        assert decision_dates.min() > split.train_end
        assert split_predictions["train_start"].eq(split.train_start).all()
        assert split_predictions["train_end"].eq(split.train_end).all()
        assert split_predictions["test_start"].eq(split.test_start).all()
        assert split_predictions["test_end"].eq(split.test_end).all()
        assert set(split_predictions["ticker"]) == {"AAA", "BBB"}

    for frame in result.values():
        assert FORBIDDEN_COLUMNS.isdisjoint(frame.columns)


def test_summaries_are_produced_per_split_and_horizon() -> None:
    result = _run(_sample_data())
    expected_pairs = {
        (split_id, horizon) for split_id in (0, 1) for horizon in HORIZONS
    }

    prediction_summary = result["prediction_summary"]
    cost_adjusted_summary = result["cost_adjusted_summary"]

    assert _split_horizon_pairs(prediction_summary) == expected_pairs
    assert _split_horizon_pairs(cost_adjusted_summary) == expected_pairs
    assert prediction_summary["no_trade_count"].sum() > 0
    assert cost_adjusted_summary["missing_cost_adjusted_count"].sum() > 0
    assert {
        "trade_count",
        "mean_signed_forward_return",
        "mean_cost_adjusted_signed_forward_return",
        "median_cost_adjusted_signed_forward_return",
        "std_cost_adjusted_signed_forward_return",
        "min_cost_adjusted_signed_forward_return",
        "max_cost_adjusted_signed_forward_return",
    }.issubset(cost_adjusted_summary.columns)


def test_custom_prediction_kwargs_are_passed_through() -> None:
    data = _sample_data()

    default_result = _run(data)
    strict_result = _run(data, prediction_kwargs={"min_score_to_trade": 4})

    assert default_result["predictions"]["trade_allowed"].sum() > 0
    assert strict_result["predictions"]["trade_allowed"].sum() == 0
    assert strict_result["prediction_summary"]["trade_count"].sum() == 0
    assert strict_result["cost_adjusted_summary"]["trade_count"].sum() == 0
    assert strict_result["cost_adjusted_summary"][
        "mean_cost_adjusted_signed_forward_return"
    ].isna().all()


def test_custom_cost_kwargs_affect_estimated_round_trip_cost_rate() -> None:
    result = _run(
        _sample_data(),
        cost_kwargs={
            "commission_bps": 2.0,
            "slippage_bps": 1.0,
            "extra_cost_bps": 6.0,
        },
    )

    costs = result["predictions"]["estimated_round_trip_cost_rate"]
    assert costs.nunique() == 1
    assert costs.iloc[0] == pytest.approx(0.0012)


def test_input_dataframe_is_not_mutated() -> None:
    data = _sample_data()
    original = data.copy(deep=True)

    _run(data)

    pd.testing.assert_frame_equal(data, original)


def test_missing_required_feature_columns_raise_value_error() -> None:
    data = _sample_data().drop(columns=["gap_pct"])

    with pytest.raises(ValueError, match="gap_pct"):
        _run(data)


def test_missing_required_label_columns_raise_value_error() -> None:
    data = _sample_data().drop(columns=["forward_return_5d"])

    with pytest.raises(ValueError, match="forward_return_5d"):
        _run(data)


def test_invalid_walk_forward_window_settings_raise_value_error() -> None:
    with pytest.raises(ValueError, match="train_window"):
        run_walk_forward_baseline_evaluation(
            _sample_data(),
            train_window=0,
            test_window=2,
            step_size=2,
        )


def test_label_date_columns_do_not_affect_predictions_or_summaries() -> None:
    data = _sample_data()
    changed = data.copy(deep=True)
    for horizon in HORIZONS:
        changed[f"label_date_{horizon}d"] = pd.date_range(
            "2040-01-01",
            periods=len(changed),
            freq="D",
        )

    baseline = _run(data)
    changed_result = _run(changed)

    assert {
        "label_date_1d",
        "label_date_3d",
        "label_date_5d",
    }.issubset(baseline["predictions"].columns)

    prediction_compare_columns = [
        column
        for column in baseline["predictions"].columns
        if "label_date" not in column
    ]
    pd.testing.assert_frame_equal(
        baseline["predictions"][prediction_compare_columns],
        changed_result["predictions"][prediction_compare_columns],
    )
    pd.testing.assert_frame_equal(
        baseline["prediction_summary"],
        changed_result["prediction_summary"],
    )
    pd.testing.assert_frame_equal(
        baseline["cost_adjusted_summary"],
        changed_result["cost_adjusted_summary"],
    )


def test_no_trade_rows_remain_represented_in_outputs() -> None:
    result = _run(_sample_data())
    predictions = result["predictions"]

    no_trade_rows = predictions[predictions["predicted_direction"].eq("no_trade")]

    assert not no_trade_rows.empty
    assert not no_trade_rows["trade_allowed"].any()
    assert no_trade_rows["no_trade_reason"].eq("weak_signal").all()
    assert result["prediction_summary"]["no_trade_count"].sum() > 0
    assert result["cost_adjusted_summary"]["missing_cost_adjusted_count"].sum() > 0
