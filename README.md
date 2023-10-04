# 2023-2024 Regular Season Point Prediction for Sour Candy Contest

(Diane's Entry)

https://hockeyviz.com/txt/predictionContest2223

## Data

The script `data.py` grabs all the data used here from the NHL API, going back to the 1995-1996 season since that's when regular seasons became 82 games in length.

If a team relocated since the start of this data, the data for that team in its original (as of 1995) location is considered part of the history for that team in its new location.  In other words, regular season points accululated by the Hartford Whalers (HFD) are part of the history of regular season points accumulated by the Carolina Hurricanes (CAR).

Since the 2003-2004 season didn't happen, point totals from the 2002-2003 season are copied over and treated as 2003-2004's point totals so as to not disrupt the time series.  (Another option could be to take the mean of 2001-2002 and 2002-2003, or even fit an autoregression on all data prior to 2003-2004...hmmm...)

## Every Team Except Seattle and Vegas

## Seattle

## Vegas

## Ranking