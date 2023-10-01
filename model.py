from datetime import date

import pandas
from prophet import Prophet


def fit_predict(team_df):
    (train_df, test_df, centring_mean, centring_std) = preprocess(team_df, True)
    eval_df = _go(train_df, test_df, centring_mean, centring_std)
    print()
    print(eval_df)

    (train_df, future_df, centring_mean, centring_std) = preprocess(team_df, False)
    forecast_df = _go(train_df, future_df, centring_mean, centring_std)
    print()
    print(forecast_df)


def _go(train_df, test_df, centring_mean, centring_std):
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                mcmc_samples=1000, interval_width=0.95, seasonality_prior_scale=30)
    m.add_regressor("lag1")
    m.add_regressor("lag2")
    m.add_regressor("lag3")
    m.fit(train_df)
    
    pred_df = m.predict(test_df)[["ds", "yhat_lower", "yhat", "yhat_upper"]]
    pred_df["ds"] = pandas.to_datetime(pred_df["ds"])
    pred_df = pred_df.merge(test_df, on=["ds"], how="left")
    for col in pred_df.columns:
        if col == "ds":
            continue
        pred_df[col] = (pred_df[col] * centring_std) + centring_mean

    return pred_df


def preprocess(team_df, model_training):
    team_df = team_df[["season_end", "total_points"]].copy()
    team_df["ds"] = team_df["season_end"].apply(lambda x : date(month=5, day=1, year=x))
    team_df = team_df.rename(columns={"total_points" : "y"})
    team_df["ds"] = pandas.to_datetime(team_df["ds"])
    del team_df["season_end"]

    if model_training:
        centring_mean = team_df["y"][0:-1].mean()
        centring_std = team_df["y"][0:-1].std()
    else:
        centring_mean = team_df["y"].mean()
        centring_std = team_df["y"].std()

    team_df["y"] = (team_df["y"] - centring_mean) / centring_std

    team_df["lag1"] = team_df["y"].shift()
    if model_training:
        team_df["lag1"] = team_df["lag1"].fillna(team_df["y"][0:-1].mean())
    else:
        team_df["lag1"] = team_df["lag1"].fillna(team_df["y"].mean())
    team_df["lag2"] = team_df["lag1"].shift().fillna(team_df["lag1"].mean())
    team_df["lag3"] = team_df["lag2"].shift().fillna(team_df["lag2"].mean())

    if model_training:
        train_df = team_df[0:-1].copy()
        test_df = team_df[-1:].copy()
        return (train_df, test_df, centring_mean, centring_std)

    forecast_df = pandas.DataFrame([{"ds" : team_df.iloc[-1]["ds"],
                                     "lag1" : team_df.iloc[-1]["y"],
                                     "lag2" : team_df.iloc[-1]["lag1"],
                                     "lag3" : team_df.iloc[-1]["lag2"]}])
    forecast_df["ds"] = forecast_df["ds"].apply(lambda x : date(year=x.year + 1, month=x.month, day=x.day))
    forecast_df["ds"] = pandas.to_datetime(forecast_df["ds"])
    return (team_df, forecast_df, centring_mean, centring_std)


if __name__ == "__main__":
    df = pandas.read_csv("NHL_API_point_totals_by_team_season_raw_1996-2003.csv")
    fit_predict(df[df["team"] == "BOS"])
