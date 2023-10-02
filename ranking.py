import pandas
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def predict_2024_ranking(input_csv="NHL_API_point_totals_by_team_season_raw_1996-2003.csv"):
    df = pandas.read_csv("NHL_API_point_totals_by_team_season_raw_1996-2003.csv")
    recent_years = sorted(set(df["season_end"]))[-6:]
    rank_df = df[df["season_end"].isin(recent_years)].pivot(index="team", columns="season_end", values="points_rank")
    rank_df = rank_df.fillna(len(rank_df))

    rank_df.columns = [f"r{i}" for i in range(0, len(rank_df.columns) - 1)] + ["y"]
    features = [c for c in rank_df.columns if c.startswith("r")]
    rank_df["y_class"] = rank_df["y"].apply(lambda x : 0 if x <= 16 else 1)

    rf = RandomForestClassifier(random_state=1024, n_jobs=-1)
    rf.fit(rank_df[features], rank_df["y_class"])
    rank_df["y_pred"] = rf.predict(rank_df[features])
    # print((rank_df["y_class"] == rank_df["y_pred"]).all())

    next_rank_df = rank_df.shift(-1, axis=1)
    del next_rank_df["y"]
    del next_rank_df["y_class"]
    del next_rank_df["y_pred"]

    # next_rank_df["rank_2024"] = rf.predict(next_rank_df[features])
    next_rank_df["prob_low_rank"] = rf.predict_proba(next_rank_df[features])[:, 1]
    return next_rank_df["prob_low_rank"].rank(method="min").sort_values()


if __name__ == "__main__":
    predict_2024_ranking()
