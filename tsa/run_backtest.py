import pandas as pd
from tsa.backtester import TSBacktester
from tsa.config import *
from models.regression import *

def main():

    print("Starting backtest...")



    # load data
    df = pd.read_csv("data/time_series.csv", index_col=0)
    df.index = pd.to_datetime(df.index)

    # generate features
    dummies = pd.get_dummies(df.index.month, prefix="month", drop_first=True)
    dummies.index = df.index
    df = pd.concat([df, dummies], axis=1)

    # run the backtest
    backtester = TSBacktester(
        MODEL_DICT[SELECTED_MODEL],
        BT_START_DATE,
        BT_END_DATE,
        BACKTEST_FREQ,
        DATA_FREQ,
        FCST_HORIZON,
        rolling_window_size=ROLLING_WINDOW_SIZE,
    )

    if SELECTED_MODEL in MODELS_W_FEATURES:
        backtester.run_backtest(df, target_col="y", verbose=True, features=FEATURE_LIST)
    else:
        backtester.run_backtest(df, target_col="y", verbose=True)

    backtest_metadata, backtest_results = backtester.evaluate_backtest(
        BT_METRICS, model_name=SELECTED_MODEL
    )

    print("Backtest results ----")
    print(backtest_results)


    for metric_name, metric_value in backtest_results.items():
        print(f"{metric_name}: {metric_value}")

        scores_df = pd.DataFrame(
            list(backtest_results.items()), columns=["metric", "value"]
        )

    for metric in METRICS_TO_PLOT:
        temp_df = scores_df[
            scores_df["metric"].str.contains(metric)
            & ~scores_df["metric"].str.contains("total")
        ]
        temp_df["horizon"] = temp_df["metric"].str.split("_").str[-1]

        # live.log_plot(
        #     f"backtest_scores_{metric}",
        #     temp_df,
        #     x="horizon",
        #     y="value",
        #     template="linear",
        #     title=f"Backtest scores - {metric}",
        #     y_label="Score",
        #     x_label="Horizon",
        # )
    print("Backtest metadata ----")

if __name__ == "__main__":
    main()