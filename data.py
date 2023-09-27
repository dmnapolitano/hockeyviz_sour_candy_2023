import requests
from tqdm import tqdm
import pandas


URL = "https://statsapi.web.nhl.com/api/v1/teams"
RELOCATED = {"WIN" : "ARI", "ATL" : "WPG", "HFD" : "CAR", "PHX" : "ARI"}


def get_one_response(season_start, season_end):
    params = {"expand" : "team.stats",
              "season" : str(season_start) + str(season_end),
              "Content-Type": "application/json"}
    response = requests.get(URL, params=params)
    response.raise_for_status()
    nhl_json = response.json()

    rows = []
    for team in nhl_json["teams"]:
        if team["abbreviation"] in RELOCATED:
            row = {"team" : RELOCATED[team["abbreviation"]]}
        else:
            row = {"team" : team["abbreviation"]}
        row["season_end"] = season_end
        # print(team["firstYearOfPlay"])
        # print(team["division"]["name"])
        # print(team["conference"]["name"])
        for team_stat in team["teamStats"]:
            if team_stat["type"]["displayName"] == "statsSingleSeason":
                for stat in team_stat["splits"]:
                    for stat_key in stat["stat"]:
                        # TODO: come back for more depending on the modelling approach
                        if stat_key == "pts":
                            if type(stat["stat"][stat_key]) is str:
                                row["points_rank"] = int(
                                    stat["stat"][stat_key].replace("rd", "").replace("st", "").replace("nd", "").replace("th", ""))
                            else:
                                row["total_points"] = stat["stat"][stat_key]
        rows.append(row)

    return pandas.DataFrame(rows)


def get_data_from_several_seasons(first_season_start=1995, first_season_end=1996):
    current_season_start = first_season_start
    current_season_end = first_season_end
    dfs = []

    with tqdm() as progress_bar:
        while current_season_end != 2024:
            if current_season_start == 2004 and current_season_end == 2005:
                current_season_start += 1
                current_season_end += 1
                continue # lockout
            this_df = get_one_response(current_season_start, current_season_end)
            dfs.append(this_df)
            if current_season_end == 2004:
                # duplicate the 2003-2004 data for the lockout season
                lockout_df = this_df.copy()
                lockout_df["season_end"] = 2005
                dfs.append(lockout_df)
            current_season_start += 1
            current_season_end += 1
            progress_bar.update(1)

    df = pandas.concat(dfs, ignore_index=True)

    print(df.groupby(["team"]).agg({"season_end" : ["nunique", "min", "max"]}).T)
    return df


if __name__ == "__main__":
    df = get_data_from_several_seasons()
    df.to_csv("NHL_API_point_totals_by_team_season_raw_1996-2003.csv", index=False)
