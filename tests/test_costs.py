import pandas as pd
import pytest

from quantv2.backtest.costs import (
    add_cost_adjusted_returns,
    calculate_round_trip_cost_rate,
)


def _sample_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC", "DDD"],
            "predicted_direction": ["up", "down", "no_trade", "up"],
            "trade_allowed": [True, True, False, True],
            "forward_return_1d": [0.10, -0.20, 0.30, None],
            "forward_return_3d": [0.20, 0.10, None, 0.05],
            "forward_return_5d": [0.05, -0.10, 0.00, None],
            "label_date_1d": pd.date_range("2030-01-01", periods=4, freq="D"),
            "label_date_3d": pd.date_range("2030-02-01", periods=4, freq="D"),
            "label_date_5d": pd.date_range("2030-03-01", periods=4, freq="D"),
            "close": [100.0, 110.0, 120.0, 130.0],
            "future_close": [1000.0, 10.0, 500.0, 1.0],
        }
    )


def _added_columns(horizons: tuple[int, ...] = (1, 3, 5)) -> set[str]:
    columns = {"estimated_round_trip_cost_rate"}
    for horizon in horizons:
        columns.add(f"signed_forward_return_{horizon}d")
        columns.add(f"cost_adjusted_signed_forward_return_{horizon}d")
    return columns


def test_calculate_round_trip_cost_rate_converts_bps_correctly() -> None:
    cost_rate = calculate_round_trip_cost_rate(
        commission_bps=1.5,
        slippage_bps=5.0,
        extra_cost_bps=2.0,
    )

    assert cost_rate == pytest.approx((2 * 1.5 + 2 * 5.0 + 2.0) / 10_000)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"commission_bps": -0.1},
        {"slippage_bps": -0.1},
        {"extra_cost_bps": -0.1},
    ],
)
def test_negative_cost_inputs_raise_value_error(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        calculate_round_trip_cost_rate(**kwargs)


def test_add_cost_adjusted_returns_preserves_rows_and_original_columns() -> None:
    data = _sample_predictions()
    result = add_cost_adjusted_returns(data)

    assert len(result) == len(data)
    assert list(result.index) == list(data.index)
    assert set(data.columns).issubset(result.columns)


def test_input_dataframe_is_not_mutated() -> None:
    data = _sample_predictions()
    original = data.copy(deep=True)

    add_cost_adjusted_returns(data)

    pd.testing.assert_frame_equal(data, original)


def test_missing_required_prediction_columns_raise_value_error() -> None:
    data = _sample_predictions().drop(columns=["predicted_direction"])

    with pytest.raises(
        ValueError,
        match="missing required prediction columns: predicted_direction",
    ):
        add_cost_adjusted_returns(data)


def test_missing_required_forward_return_columns_raise_value_error() -> None:
    data = _sample_predictions().drop(columns=["forward_return_5d"])

    with pytest.raises(
        ValueError,
        match="missing required forward-return columns: forward_return_5d",
    ):
        add_cost_adjusted_returns(data)


def test_invalid_predicted_direction_raises_value_error() -> None:
    data = _sample_predictions()
    data.loc[0, "predicted_direction"] = "sideways"

    with pytest.raises(ValueError, match="predicted_direction contains invalid values"):
        add_cost_adjusted_returns(data)


def test_invalid_trade_allowed_value_raises_value_error() -> None:
    data = _sample_predictions()
    data["trade_allowed"] = data["trade_allowed"].astype("object")
    data.loc[0, "trade_allowed"] = "maybe"

    with pytest.raises(ValueError, match="trade_allowed must be boolean-like"):
        add_cost_adjusted_returns(data)


def test_trade_allowed_true_with_no_trade_prediction_raises_value_error() -> None:
    data = _sample_predictions()
    data.loc[0, "predicted_direction"] = "no_trade"

    with pytest.raises(
        ValueError,
        match="trade_allowed=True requires predicted_direction to be up or down",
    ):
        add_cost_adjusted_returns(data)


def test_predicted_direction_no_trade_with_trade_allowed_false_is_valid() -> None:
    data = pd.DataFrame(
        {
            "predicted_direction": ["no_trade"],
            "trade_allowed": [False],
            "forward_return_1d": [0.10],
        }
    )

    result = add_cost_adjusted_returns(data, horizons=(1,))

    assert pd.isna(result.loc[0, "signed_forward_return_1d"])
    assert pd.isna(result.loc[0, "cost_adjusted_signed_forward_return_1d"])


def test_signed_forward_return_is_correct_for_up_predictions() -> None:
    result = add_cost_adjusted_returns(_sample_predictions())

    assert result.loc[0, "signed_forward_return_1d"] == pytest.approx(0.10)
    assert result.loc[0, "signed_forward_return_3d"] == pytest.approx(0.20)
    assert result.loc[0, "signed_forward_return_5d"] == pytest.approx(0.05)


def test_signed_forward_return_is_correct_for_down_predictions() -> None:
    result = add_cost_adjusted_returns(_sample_predictions())

    assert result.loc[1, "signed_forward_return_1d"] == pytest.approx(0.20)
    assert result.loc[1, "signed_forward_return_3d"] == pytest.approx(-0.10)
    assert result.loc[1, "signed_forward_return_5d"] == pytest.approx(0.10)


def test_no_trade_rows_get_nan_signed_and_cost_adjusted_returns() -> None:
    result = add_cost_adjusted_returns(_sample_predictions())

    assert pd.isna(result.loc[2, "signed_forward_return_1d"])
    assert pd.isna(result.loc[2, "cost_adjusted_signed_forward_return_1d"])
    assert pd.isna(result.loc[2, "signed_forward_return_3d"])
    assert pd.isna(result.loc[2, "cost_adjusted_signed_forward_return_3d"])
    assert pd.isna(result.loc[2, "signed_forward_return_5d"])
    assert pd.isna(result.loc[2, "cost_adjusted_signed_forward_return_5d"])


def test_missing_forward_return_labels_get_nan_signed_and_cost_adjusted_returns() -> None:
    result = add_cost_adjusted_returns(_sample_predictions())

    assert pd.isna(result.loc[3, "signed_forward_return_1d"])
    assert pd.isna(result.loc[3, "cost_adjusted_signed_forward_return_1d"])
    assert result.loc[3, "signed_forward_return_3d"] == pytest.approx(0.05)
    assert pd.isna(result.loc[3, "signed_forward_return_5d"])
    assert pd.isna(result.loc[3, "cost_adjusted_signed_forward_return_5d"])


def test_cost_adjusted_signed_forward_return_subtracts_cost_rate() -> None:
    cost_rate = calculate_round_trip_cost_rate(
        commission_bps=1.0,
        slippage_bps=2.0,
        extra_cost_bps=3.0,
    )
    result = add_cost_adjusted_returns(
        _sample_predictions(),
        commission_bps=1.0,
        slippage_bps=2.0,
        extra_cost_bps=3.0,
    )

    assert result.loc[0, "cost_adjusted_signed_forward_return_1d"] == pytest.approx(
        result.loc[0, "signed_forward_return_1d"] - cost_rate
    )
    assert result.loc[1, "cost_adjusted_signed_forward_return_5d"] == pytest.approx(
        result.loc[1, "signed_forward_return_5d"] - cost_rate
    )


def test_estimated_round_trip_cost_rate_is_added_and_constant_across_rows() -> None:
    result = add_cost_adjusted_returns(
        _sample_predictions(),
        commission_bps=1.0,
        slippage_bps=2.0,
        extra_cost_bps=3.0,
    )

    assert result["estimated_round_trip_cost_rate"].nunique() == 1
    assert result["estimated_round_trip_cost_rate"].iloc[0] == pytest.approx(0.0009)


def test_label_date_columns_are_ignored() -> None:
    result = add_cost_adjusted_returns(_sample_predictions())

    assert "label_date_1d" in result.columns
    assert "label_date_3d" in result.columns
    assert "label_date_5d" in result.columns


def test_changing_label_date_columns_does_not_change_cost_adjusted_outputs() -> None:
    data = _sample_predictions()
    changed = data.copy(deep=True)
    changed["label_date_1d"] = pd.date_range("2040-01-01", periods=4, freq="D")
    changed["label_date_3d"] = pd.date_range("2040-02-01", periods=4, freq="D")
    changed["label_date_5d"] = pd.date_range("2040-03-01", periods=4, freq="D")

    baseline = add_cost_adjusted_returns(data)
    changed_result = add_cost_adjusted_returns(changed)
    output_columns = sorted(_added_columns())

    pd.testing.assert_frame_equal(
        baseline[output_columns],
        changed_result[output_columns],
    )


def test_price_columns_are_not_used_to_generate_labels() -> None:
    data = pd.DataFrame(
        {
            "predicted_direction": ["up"],
            "trade_allowed": [True],
            "forward_return_1d": [None],
            "close": [100.0],
            "future_close": [200.0],
        }
    )

    result = add_cost_adjusted_returns(data, horizons=(1,))

    assert pd.isna(result.loc[0, "signed_forward_return_1d"])
    assert pd.isna(result.loc[0, "cost_adjusted_signed_forward_return_1d"])


def test_function_does_not_create_execution_pnl_profit_or_trade_columns() -> None:
    data = _sample_predictions()
    result = add_cost_adjusted_returns(data)
    new_columns = set(result.columns) - set(data.columns)
    restricted_columns = {
        "pnl",
        "profit",
        "position",
        "order",
        "execution",
        "fill",
        "brokerage",
        "trade",
    }

    assert restricted_columns.isdisjoint(new_columns)


def test_all_requested_horizons_are_handled() -> None:
    result = add_cost_adjusted_returns(_sample_predictions(), horizons=(1, 3, 5))

    assert _added_columns().issubset(result.columns)


def test_custom_horizons_work_when_matching_forward_return_columns_exist() -> None:
    data = pd.DataFrame(
        {
            "predicted_direction": ["up", "down"],
            "trade_allowed": [True, True],
            "forward_return_2d": [0.04, 0.03],
            "forward_return_10d": [-0.05, -0.06],
        }
    )

    result = add_cost_adjusted_returns(data, horizons=(2, 10), slippage_bps=0.0)

    assert result.loc[0, "signed_forward_return_2d"] == pytest.approx(0.04)
    assert result.loc[1, "signed_forward_return_2d"] == pytest.approx(-0.03)
    assert result.loc[0, "signed_forward_return_10d"] == pytest.approx(-0.05)
    assert result.loc[1, "signed_forward_return_10d"] == pytest.approx(0.06)
    assert result.loc[0, "cost_adjusted_signed_forward_return_2d"] == pytest.approx(
        0.04
    )


def test_missing_custom_horizon_label_columns_raise_value_error() -> None:
    data = _sample_predictions()

    with pytest.raises(
        ValueError,
        match="missing required forward-return columns: forward_return_10d",
    ):
        add_cost_adjusted_returns(data, horizons=(1, 10))
