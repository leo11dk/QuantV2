import pandas as pd
import pytest

from quantv2.evaluation.prediction_metrics import evaluate_baseline_predictions


EXPECTED_COLUMNS = [
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


def _sample_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
            "event_type": [
                "earnings",
                "guidance",
                "earnings",
                "guidance",
                "earnings",
                "guidance",
            ],
            "predicted_direction": ["up", "down", "up", "down", "no_trade", "up"],
            "trade_allowed": [True, True, True, True, False, False],
            "no_trade_reason": [
                "trade_allowed",
                "trade_allowed",
                "trade_allowed",
                "trade_allowed",
                "weak_signal",
                "high_volatility",
            ],
            "forward_return_1d": [0.10, -0.20, -0.05, 0.00, 0.30, None],
            "forward_return_3d": [0.20, 0.10, -0.10, 0.05, None, -0.02],
            "forward_return_5d": [0.05, -0.10, 0.02, -0.03, 0.00, None],
            "label_date_1d": pd.date_range("2030-01-01", periods=6, freq="D"),
            "label_date_3d": pd.date_range("2030-02-01", periods=6, freq="D"),
            "close": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            "future_close": [1000.0, 10.0, 500.0, 1.0, 300.0, 200.0],
        }
    )


def test_overall_prediction_evaluation_works() -> None:
    summary = evaluate_baseline_predictions(_sample_predictions())

    assert summary.columns.tolist() == EXPECTED_COLUMNS
    assert summary["horizon"].tolist() == [1, 3, 5]

    one_day = summary.loc[summary["horizon"].eq(1)].iloc[0]
    assert one_day["total_count"] == 6
    assert one_day["evaluated_count"] == 5
    assert one_day["missing_label_count"] == 1
    assert one_day["trade_count"] == 4
    assert one_day["no_trade_count"] == 1
    assert one_day["coverage"] == pytest.approx(4 / 5)
    assert one_day["hit_count"] == 2
    assert one_day["hit_rate"] == pytest.approx(2 / 4)
    assert one_day["mean_signed_forward_return"] == pytest.approx(0.0625)
    assert one_day["median_signed_forward_return"] == pytest.approx(0.05)
    assert one_day["std_signed_forward_return"] == pytest.approx(
        pd.Series([0.10, 0.20, -0.05, -0.00]).std()
    )
    assert one_day["mean_forward_return_all"] == pytest.approx(0.03)
    assert one_day["mean_forward_return_traded"] == pytest.approx(-0.0375)


def test_grouped_evaluation_works_with_one_group_column() -> None:
    summary = evaluate_baseline_predictions(
        _sample_predictions(),
        group_cols=("ticker",),
    )

    assert summary.columns.tolist() == ["ticker", *EXPECTED_COLUMNS]

    aaa_one_day = summary[
        summary["ticker"].eq("AAA") & summary["horizon"].eq(1)
    ].iloc[0]
    assert aaa_one_day["total_count"] == 2
    assert aaa_one_day["evaluated_count"] == 2
    assert aaa_one_day["trade_count"] == 2
    assert aaa_one_day["coverage"] == pytest.approx(1.0)
    assert aaa_one_day["hit_rate"] == pytest.approx(1.0)


def test_grouped_evaluation_works_with_multiple_group_columns() -> None:
    summary = evaluate_baseline_predictions(
        _sample_predictions(),
        group_cols=("ticker", "event_type"),
        horizons=(1,),
    )

    assert summary.columns.tolist() == ["ticker", "event_type", *EXPECTED_COLUMNS]

    bbb_guidance = summary[
        summary["ticker"].eq("BBB")
        & summary["event_type"].eq("guidance")
        & summary["horizon"].eq(1)
    ].iloc[0]
    assert bbb_guidance["total_count"] == 1
    assert bbb_guidance["trade_count"] == 1
    assert bbb_guidance["hit_count"] == 0
    assert bbb_guidance["hit_rate"] == pytest.approx(0.0)


def test_directional_hit_rate_is_correct_for_up_and_down_predictions() -> None:
    summary = evaluate_baseline_predictions(_sample_predictions(), horizons=(1,))

    one_day = summary.iloc[0]
    assert one_day["hit_count"] == 2
    assert one_day["hit_rate"] == pytest.approx(0.5)


def test_zero_forward_returns_are_not_counted_as_correct() -> None:
    data = pd.DataFrame(
        {
            "predicted_direction": ["up", "down"],
            "trade_allowed": [True, True],
            "no_trade_reason": ["trade_allowed", "trade_allowed"],
            "forward_return_1d": [0.0, 0.0],
        }
    )

    summary = evaluate_baseline_predictions(data, horizons=(1,))

    assert summary.loc[0, "hit_count"] == 0
    assert summary.loc[0, "hit_rate"] == pytest.approx(0.0)


def test_coverage_is_calculated_correctly() -> None:
    summary = evaluate_baseline_predictions(_sample_predictions(), horizons=(1,))

    assert summary.loc[0, "coverage"] == pytest.approx(4 / 5)


def test_trade_count_and_no_trade_count_are_calculated_correctly() -> None:
    summary = evaluate_baseline_predictions(_sample_predictions(), horizons=(1,))

    assert summary.loc[0, "trade_count"] == 4
    assert summary.loc[0, "no_trade_count"] == 1


def test_missing_labels_are_counted_and_excluded_from_directional_evaluation() -> None:
    data = pd.DataFrame(
        {
            "predicted_direction": ["up", "down"],
            "trade_allowed": [True, True],
            "no_trade_reason": ["trade_allowed", "trade_allowed"],
            "forward_return_1d": [None, -0.10],
        }
    )

    summary = evaluate_baseline_predictions(data, horizons=(1,))

    assert summary.loc[0, "total_count"] == 2
    assert summary.loc[0, "evaluated_count"] == 1
    assert summary.loc[0, "missing_label_count"] == 1
    assert summary.loc[0, "trade_count"] == 1
    assert summary.loc[0, "hit_count"] == 1
    assert summary.loc[0, "hit_rate"] == pytest.approx(1.0)


def test_no_trade_rows_are_counted_but_excluded_from_hit_rate() -> None:
    data = pd.DataFrame(
        {
            "predicted_direction": ["up", "no_trade"],
            "trade_allowed": [True, False],
            "no_trade_reason": ["trade_allowed", "weak_signal"],
            "forward_return_1d": [0.10, -0.20],
        }
    )

    summary = evaluate_baseline_predictions(data, horizons=(1,))

    assert summary.loc[0, "trade_count"] == 1
    assert summary.loc[0, "no_trade_count"] == 1
    assert summary.loc[0, "hit_count"] == 1
    assert summary.loc[0, "hit_rate"] == pytest.approx(1.0)


def test_no_traded_rows_produce_nan_hit_rate_and_signed_return_summaries() -> None:
    data = pd.DataFrame(
        {
            "predicted_direction": ["no_trade", "up"],
            "trade_allowed": [False, False],
            "no_trade_reason": ["weak_signal", "high_volatility"],
            "forward_return_1d": [0.10, -0.20],
        }
    )

    summary = evaluate_baseline_predictions(data, horizons=(1,))

    assert summary.loc[0, "trade_count"] == 0
    assert summary.loc[0, "no_trade_count"] == 2
    assert summary.loc[0, "coverage"] == pytest.approx(0.0)
    assert pd.isna(summary.loc[0, "hit_rate"])
    assert pd.isna(summary.loc[0, "mean_signed_forward_return"])
    assert pd.isna(summary.loc[0, "median_signed_forward_return"])
    assert pd.isna(summary.loc[0, "std_signed_forward_return"])


def test_all_missing_labels_produce_safe_nan_metrics() -> None:
    data = pd.DataFrame(
        {
            "predicted_direction": ["up", "down"],
            "trade_allowed": [True, True],
            "no_trade_reason": ["trade_allowed", "trade_allowed"],
            "forward_return_1d": [None, None],
        }
    )

    summary = evaluate_baseline_predictions(data, horizons=(1,))

    assert summary.loc[0, "evaluated_count"] == 0
    assert summary.loc[0, "missing_label_count"] == 2
    assert summary.loc[0, "trade_count"] == 0
    assert summary.loc[0, "no_trade_count"] == 0
    assert pd.isna(summary.loc[0, "coverage"])
    assert pd.isna(summary.loc[0, "hit_rate"])
    assert pd.isna(summary.loc[0, "mean_signed_forward_return"])
    assert pd.isna(summary.loc[0, "mean_forward_return_all"])
    assert pd.isna(summary.loc[0, "mean_forward_return_traded"])


def test_input_dataframe_is_not_mutated() -> None:
    data = _sample_predictions()
    original = data.copy(deep=True)

    evaluate_baseline_predictions(data)

    pd.testing.assert_frame_equal(data, original)


def test_missing_required_prediction_columns_raise_value_error() -> None:
    data = _sample_predictions().drop(columns=["predicted_direction"])

    with pytest.raises(
        ValueError,
        match="missing required prediction columns: predicted_direction",
    ):
        evaluate_baseline_predictions(data)


def test_missing_forward_return_columns_raise_value_error() -> None:
    data = _sample_predictions().drop(columns=["forward_return_5d"])

    with pytest.raises(
        ValueError,
        match="missing required forward-return columns: forward_return_5d",
    ):
        evaluate_baseline_predictions(data)


def test_missing_group_columns_raise_value_error() -> None:
    with pytest.raises(ValueError, match="missing group columns: sector"):
        evaluate_baseline_predictions(_sample_predictions(), group_cols=("sector",))


def test_invalid_predicted_direction_values_raise_value_error() -> None:
    data = _sample_predictions()
    data.loc[0, "predicted_direction"] = "sideways"

    with pytest.raises(ValueError, match="predicted_direction contains invalid values"):
        evaluate_baseline_predictions(data)


def test_trade_allowed_true_with_no_trade_prediction_raises_value_error() -> None:
    data = _sample_predictions()
    data.loc[0, "predicted_direction"] = "no_trade"

    with pytest.raises(
        ValueError,
        match="trade_allowed=True requires predicted_direction to be up or down",
    ):
        evaluate_baseline_predictions(data)


def test_up_or_down_prediction_with_trade_allowed_false_is_counted_as_no_trade() -> None:
    data = pd.DataFrame(
        {
            "predicted_direction": ["up", "down"],
            "trade_allowed": [False, False],
            "no_trade_reason": ["high_volatility", "weak_signal"],
            "forward_return_1d": [0.10, -0.20],
        }
    )

    summary = evaluate_baseline_predictions(data, horizons=(1,))

    assert summary.loc[0, "trade_count"] == 0
    assert summary.loc[0, "no_trade_count"] == 2
    assert summary.loc[0, "coverage"] == pytest.approx(0.0)


def test_label_date_columns_are_ignored() -> None:
    summary = evaluate_baseline_predictions(_sample_predictions(), horizons=(1,))

    assert "label_date_1d" not in summary.columns
    assert "label_date_3d" not in summary.columns


def test_changing_label_date_columns_does_not_change_evaluation() -> None:
    data = _sample_predictions()
    changed = data.copy(deep=True)
    changed["label_date_1d"] = pd.date_range("2040-01-01", periods=6, freq="D")
    changed["label_date_3d"] = pd.date_range("2040-02-01", periods=6, freq="D")

    baseline = evaluate_baseline_predictions(data)
    changed_summary = evaluate_baseline_predictions(changed)

    pd.testing.assert_frame_equal(baseline, changed_summary)


def test_label_date_columns_cannot_be_used_as_group_columns() -> None:
    with pytest.raises(
        ValueError,
        match="group_cols cannot include label_date columns: label_date_1d",
    ):
        evaluate_baseline_predictions(
            _sample_predictions(),
            group_cols=("label_date_1d",),
        )


def test_function_does_not_create_order_execution_position_pnl_or_profit_columns() -> None:
    data = _sample_predictions()
    original_columns = set(data.columns)

    summary = evaluate_baseline_predictions(data)

    restricted_columns = {"order", "execution", "position", "pnl", "profit"}
    assert restricted_columns.isdisjoint(summary.columns)
    assert set(data.columns) == original_columns


def test_function_does_not_generate_labels_from_price_columns() -> None:
    data = pd.DataFrame(
        {
            "predicted_direction": ["up"],
            "trade_allowed": [True],
            "no_trade_reason": ["trade_allowed"],
            "forward_return_1d": [None],
            "close": [100.0],
            "future_close": [200.0],
        }
    )

    summary = evaluate_baseline_predictions(data, horizons=(1,))

    assert summary.loc[0, "total_count"] == 1
    assert summary.loc[0, "evaluated_count"] == 0
    assert summary.loc[0, "missing_label_count"] == 1
    assert pd.isna(summary.loc[0, "mean_forward_return_all"])
