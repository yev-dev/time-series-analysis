import pytest
import pandas as pd
import numpy as np
from tsa.backtester import TSBacktester

@pytest.fixture
def sample_data():
    # Create a simple time series dataframe
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "y": np.arange(30) + np.random.normal(0, 1, 30),
        "feature1": np.random.normal(10, 2, 30),
        "feature2": np.random.normal(5, 1, 30),
    }, index=dates)
    return df

def dummy_pred_func(X_train, y_train, X_test, forecast_horizon, features=None):
    # Predicts the last value of y_train for all test points
    return np.repeat(y_train.iloc[-1], len(X_test))

def mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def test_run_backtest_expanding(sample_data):
    bt = TSBacktester(
        pred_func=dummy_pred_func,
        start_date="2023-01-10",
        end_date="2023-01-20",
        backtest_freq="5D",
        data_freq="D",
        forecast_horizon=3,
        rolling_window_size=None,
    )
    bt.run_backtest(sample_data, target_col="y", features=["feature1", "feature2"])
    assert bt.backtest_df is not None
    # Check columns
    expected_cols = {"forecast_date", "report_date", "forecast", "actual", "horizon"}
    assert expected_cols.issubset(set(bt.backtest_df.columns))
    # Check that forecast and actual are numeric
    assert np.issubdtype(bt.backtest_df["forecast"].dtype, np.number)
    assert np.issubdtype(bt.backtest_df["actual"].dtype, np.number)

def test_run_backtest_rolling(sample_data):
    bt = TSBacktester(
        pred_func=dummy_pred_func,
        start_date="2023-01-10",
        end_date="2023-01-20",
        backtest_freq="5D",
        data_freq="D",
        forecast_horizon=2,
        rolling_window_size=10,
    )
    bt.run_backtest(sample_data, target_col="y")
    assert bt.backtest_df is not None
    # Check rolling window size is respected
    # (not directly, but at least the code runs and produces output)
    assert len(bt.backtest_df) > 0

def test_evaluate_backtest_no_metrics(sample_data):
    bt = TSBacktester(
        pred_func=dummy_pred_func,
        start_date="2023-01-10",
        end_date="2023-01-20",
        backtest_freq="5D",
        data_freq="D",
        forecast_horizon=2,
    )
    bt.run_backtest(sample_data, target_col="y")
    metadata, scores = bt.evaluate_backtest({}, model_name="dummy")
    assert isinstance(metadata, dict)
    assert isinstance(scores, dict)
    assert "mae_total" not in scores  # No metrics provided, so no scores should be calculated
    assert metadata["model_name"] == "dummy"
    assert metadata["validation_type"] == "expanding"

def test_evaluate_backtest(sample_data):
    bt = TSBacktester(
        pred_func=dummy_pred_func,
        start_date="2023-01-10",
        end_date="2023-01-20",
        backtest_freq="5D",
        data_freq="D",
        forecast_horizon=2,
    )
    bt.run_backtest(sample_data, target_col="y")
    metrics = {"mae": mae}
    metadata, scores = bt.evaluate_backtest(metrics, model_name="dummy")
    assert isinstance(metadata, dict)
    assert isinstance(scores, dict)
    assert "mae_total" in scores
    assert metadata["model_name"] == "dummy"
    assert metadata["validation_type"] == "expanding"

def test_evaluate_backtest_all_metrics(sample_data):

    bt = TSBacktester(
        pred_func=dummy_pred_func,
        start_date="2023-01-10",
        end_date="2023-01-20",
        backtest_freq="5D",
        data_freq="D",
        forecast_horizon=2,
    )
    bt.run_backtest(sample_data, target_col="y")
    metrics = {"mae": mae, "mse": lambda y_true, y_pred: np.mean((np.array(y_true) - np.array(y_pred))**2)}
    metadata, scores = bt.evaluate_backtest(metrics, model_name="dummy")
    assert isinstance(metadata, dict)
    assert isinstance(scores, dict)
    assert "mae_total" in scores
    assert "mse_total" in scores
    assert metadata["model_name"] == "dummy"
    assert metadata["validation_type"] == "expanding"

def test_evaluate_backtest_raises_without_run(sample_data):
    bt = TSBacktester(
        pred_func=dummy_pred_func,
        start_date="2023-01-10",
        end_date="2023-01-20",
        backtest_freq="5D",
        data_freq="D",
        forecast_horizon=2,
    )
    metrics = {"mae": mae}
    with pytest.raises(ValueError, match="Backtest was not yet executed"):
        bt.evaluate_backtest(metrics, model_name="dummy")