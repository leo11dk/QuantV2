import pandas as pd
import pytest

from quantv2.models.rule_baseline import generate_rule_baseline_predictions


PREDICTION_COLUMNS = [
    "rule_score",
    "predicted_direction",
    "trade_allowed",
    "no_trade_reason",
]


def _sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA"],
            "decision_date": [pd.Timestamp("2024-01-02")],
            "gap_pct": [0.02],
            "prior_5d_return": [0.03],
            "volatility_20d": [0.10],
        }
    )


def _prediction_values(result: pd.DataFrame) -> pd.DataFrame:
    return result[PREDICTION_COLUMNS].reset_index(drop=True)


def test_positive_gap_and_positive_momentum_create_positive_rule_score() -> None:
    result = generate_rule_baseline_predictions(_sample_data())
    row = result.iloc[0]

    assert row["rule_score"] == 2
    assert row["predicted_direction"] == "up"
    assert row["trade_allowed"]
    assert row["no_trade_reason"] == "trade_allowed"


def test_negative_gap_and_negative_momentum_create_negative_rule_score() -> None:
    data = _sample_data()
    data["gap_pct"] = [-0.02]
    data["prior_5d_return"] = [-0.03]

    result = generate_rule_baseline_predictions(data)
    row = result.iloc[0]

    assert row["rule_score"] == -2
    assert row["predicted_direction"] == "down"
    assert row["trade_allowed"]
    assert row["no_trade_reason"] == "trade_allowed"


def test_weak_signals_become_no_trade() -> None:
    data = _sample_data()
    data["gap_pct"] = [0.00]
    data["prior_5d_return"] = [0.00]

    result = generate_rule_baseline_predictions(data)
    row = result.iloc[0]

    assert row["rule_score"] == 0
    assert row["predicted_direction"] == "no_trade"
    assert not row["trade_allowed"]
    assert row["no_trade_reason"] == "weak_signal"


def test_missing_required_features_become_no_trade_with_missing_features() -> None:
    data = _sample_data()
    data["gap_pct"] = [None]

    result = generate_rule_baseline_predictions(data)
    row = result.iloc[0]

    assert pd.isna(row["rule_score"])
    assert row["predicted_direction"] == "no_trade"
    assert not row["trade_allowed"]
    assert row["no_trade_reason"] == "missing_features"


def test_event_direction_adds_positive_score_when_positive() -> None:
    data = _sample_data()
    data["gap_pct"] = [0.00]
    data["prior_5d_return"] = [0.00]
    data["event_direction"] = ["Beat"]

    result = generate_rule_baseline_predictions(data)

    assert result.loc[0, "rule_score"] == 1
    assert result.loc[0, "predicted_direction"] == "up"


def test_event_direction_subtracts_score_when_negative() -> None:
    data = _sample_data()
    data["gap_pct"] = [0.00]
    data["prior_5d_return"] = [0.00]
    data["event_direction"] = ["DownGrade"]

    result = generate_rule_baseline_predictions(data)

    assert result.loc[0, "rule_score"] == -1
    assert result.loc[0, "predicted_direction"] == "down"


def test_event_direction_is_optional() -> None:
    data = _sample_data().drop(columns=["volatility_20d"])

    result = generate_rule_baseline_predictions(data)

    assert result.loc[0, "rule_score"] == 2
    assert result.loc[0, "predicted_direction"] == "up"


def test_max_volatility_blocks_trades_when_volatility_is_too_high() -> None:
    data = _sample_data()
    data["volatility_20d"] = [0.50]

    result = generate_rule_baseline_predictions(data, max_volatility=0.20)
    row = result.iloc[0]

    assert row["rule_score"] == 2
    assert row["predicted_direction"] == "up"
    assert not row["trade_allowed"]
    assert row["no_trade_reason"] == "high_volatility"


def test_max_volatility_requires_volatility_column() -> None:
    data = _sample_data().drop(columns=["volatility_20d"])

    with pytest.raises(ValueError, match="volatility_20d"):
        generate_rule_baseline_predictions(data, max_volatility=0.20)


def test_input_dataframe_is_not_mutated() -> None:
    data = _sample_data()
    original = data.copy(deep=True)

    generate_rule_baseline_predictions(data)

    pd.testing.assert_frame_equal(data, original)
    for column in PREDICTION_COLUMNS:
        assert column not in data.columns


def test_output_is_sorted_by_ticker_and_decision_date() -> None:
    data = pd.DataFrame(
        {
            "ticker": ["BBB", "AAA", "AAA", "BBB"],
            "decision_date": pd.to_datetime(
                ["2024-01-03", "2024-01-02", "2024-01-01", "2024-01-01"]
            ),
            "gap_pct": [0.02, 0.02, -0.02, -0.02],
            "prior_5d_return": [0.03, 0.03, -0.03, -0.03],
        }
    )

    result = generate_rule_baseline_predictions(data)

    assert result["ticker"].tolist() == ["AAA", "AAA", "BBB", "BBB"]
    assert result["decision_date"].tolist() == list(
        pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-03"])
    )


def test_forward_return_label_columns_are_ignored() -> None:
    data = _sample_data()
    data["forward_return_1d"] = [10.0]
    changed_label = data.copy(deep=True)
    changed_label["forward_return_1d"] = [-10.0]

    baseline = generate_rule_baseline_predictions(data)
    changed = generate_rule_baseline_predictions(changed_label)

    pd.testing.assert_frame_equal(
        _prediction_values(baseline),
        _prediction_values(changed),
    )


def test_label_date_columns_are_ignored() -> None:
    data = _sample_data()
    data["label_date_1d"] = [pd.Timestamp("2030-01-01")]
    changed_label = data.copy(deep=True)
    changed_label["label_date_1d"] = [pd.Timestamp("2035-01-01")]

    baseline = generate_rule_baseline_predictions(data)
    changed = generate_rule_baseline_predictions(changed_label)

    pd.testing.assert_frame_equal(
        _prediction_values(baseline),
        _prediction_values(changed),
    )


def test_future_close_label_columns_are_ignored() -> None:
    data = _sample_data()
    data["close_1d"] = [1000.0]
    changed_label = data.copy(deep=True)
    changed_label["close_1d"] = [1.0]

    baseline = generate_rule_baseline_predictions(data)
    changed = generate_rule_baseline_predictions(changed_label)

    pd.testing.assert_frame_equal(
        _prediction_values(baseline),
        _prediction_values(changed),
    )


@pytest.mark.parametrize(
    "missing_column",
    ["ticker", "decision_date", "gap_pct", "prior_5d_return"],
)
def test_missing_required_columns_raise_value_error(missing_column: str) -> None:
    data = _sample_data().drop(columns=[missing_column])

    with pytest.raises(ValueError, match=missing_column):
        generate_rule_baseline_predictions(data)


def test_function_does_not_create_label_trade_or_execution_columns() -> None:
    data = _sample_data()

    result = generate_rule_baseline_predictions(data)

    new_columns = set(result.columns).difference(data.columns)
    assert new_columns == set(PREDICTION_COLUMNS)
    assert not new_columns.intersection(
        {
            "label",
            "trade",
            "position",
            "pnl",
            "profit",
            "order",
            "execution",
        }
    )


def test_no_trade_reason_is_correct_for_all_outcomes() -> None:
    data = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA", "AAA"],
            "decision_date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "gap_pct": [0.02, 0.00, 0.02, None],
            "prior_5d_return": [0.03, 0.00, 0.03, 0.03],
            "volatility_20d": [0.10, 0.10, 0.50, 0.10],
        }
    )

    result = generate_rule_baseline_predictions(data, max_volatility=0.20)

    assert result["no_trade_reason"].tolist() == [
        "trade_allowed",
        "weak_signal",
        "high_volatility",
        "missing_features",
    ]
    assert result["trade_allowed"].tolist() == [True, False, False, False]


def test_multiple_tickers_are_handled_independently() -> None:
    data = pd.DataFrame(
        {
            "ticker": ["BBB", "AAA", "BBB", "AAA"],
            "decision_date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]
            ),
            "gap_pct": [-0.02, 0.02, 0.00, 0.00],
            "prior_5d_return": [-0.03, 0.03, 0.03, -0.03],
        }
    )

    result = generate_rule_baseline_predictions(data)

    assert result[["ticker", "decision_date", "rule_score"]].to_dict("records") == [
        {
            "ticker": "AAA",
            "decision_date": pd.Timestamp("2024-01-01"),
            "rule_score": 2.0,
        },
        {
            "ticker": "AAA",
            "decision_date": pd.Timestamp("2024-01-02"),
            "rule_score": -1.0,
        },
        {
            "ticker": "BBB",
            "decision_date": pd.Timestamp("2024-01-01"),
            "rule_score": -2.0,
        },
        {
            "ticker": "BBB",
            "decision_date": pd.Timestamp("2024-01-02"),
            "rule_score": 1.0,
        },
    ]
