from datetime import date

import pandas
from prophet import Prophet

from ranking import predict_2024_ranking


def fit_predict(team_df, rank_2024, lags=3):
    (train_df, test_df) = _preprocess(team_df, True, lags)
    eval_df = _go(train_df, test_df)

    (train_df, future_df) = _preprocess(team_df, False, lags, rank_2024=rank_2024)
    forecast_df = _go(train_df, future_df)

    return (eval_df, forecast_df)


def _go(train_df, test_df):
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                mcmc_samples=1000, interval_width=0.95, seasonality_prior_scale=30)
    m.add_regressor("points_rank")
    for col in train_df.columns:
        if col.startswith("lag"):
            m.add_regressor(col)
    m.fit(train_df, seed=1024)
    
    pred_df = m.predict(test_df)[["ds", "yhat_lower", "yhat", "yhat_upper"]]
    pred_df["ds"] = pandas.to_datetime(pred_df["ds"])
    pred_df = pred_df.merge(test_df, on=["ds"], how="left")

    return pred_df


def _preprocess(team_df, model_training, lags, rank_2024=None):
    team_df = team_df[["season_end", "total_points", "points_rank"]].copy()
    team_df["ds"] = team_df["season_end"].apply(lambda x : date(month=5, day=1, year=x))
    team_df = team_df.rename(columns={"total_points" : "y"})
    team_df["ds"] = pandas.to_datetime(team_df["ds"])
    del team_df["season_end"]

    team_df["lag1"] = team_df["y"].shift()
    if model_training:
        team_df["lag1"] = team_df["lag1"].fillna(team_df["y"][0:-1].mean())
    else:
        team_df["lag1"] = team_df["lag1"].fillna(team_df["y"].mean())
    for i in range(1, lags):
        team_df[f"lag{i + 1}"] = team_df[f"lag{i}"].shift().fillna(team_df[f"lag{i}"].mean())

    if model_training:
        train_df = team_df[0:-1].copy()
        test_df = team_df[-1:].copy()
        return (train_df, test_df)

    forecast_df = pandas.DataFrame([{"ds" : team_df.iloc[-1]["ds"],
                                     "points_rank" : rank_2024,
                                     "lag1" : team_df.iloc[-1]["y"]}])
    for i in range(1, lags):
        forecast_df[f"lag{i + 1}"] = team_df.iloc[-1][f"lag{i}"]
    forecast_df["ds"] = forecast_df["ds"].apply(lambda x : date(year=x.year + 1, month=x.month, day=x.day))
    forecast_df["ds"] = pandas.to_datetime(forecast_df["ds"])
    return (team_df, forecast_df)
