# 2023-2024 Regular Season Point Prediction for Sour Candy Contest

(Diane's Unsubmitted Entry :disappointed:)

https://hockeyviz.com/txt/predictionContest2223

## Data

The script `data.py` grabs all the data used here from the NHL API, going back to the 1995-1996 season since that's when regular seasons became 82 games in length.

If a team relocated since the start of this data, the data for that team in its original (as of 1995) location is considered part of the history for that team in its new location.  In other words, regular season points accumulated by the Hartford Whalers (HFD) are part of the history of regular season points accumulated by the Carolina Hurricanes (CAR).

Since the 2003-2004 season didn't happen, point totals from the 2002-2003 season are copied over and treated as 2003-2004's point totals so as to not disrupt the time series.  (Another option could be to take the mean of 2001-2002 and 2002-2003, or even fit an autoregression on all data prior to 2003-2004...hmmm...)

## Ranking 2024

It became clear to me that using the team's overall ranking across the entire NHL for each season `s` was a valuable feature.  This meant I needed some way to predict what each team's possible 2023-2024 ranking might be, in order to generate forecasts for that yet-to-occur season.

I trained a [random forest classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) on whether or not the most-recent team's ranking was <= 16 (0 or "good") or > 16 (1 or "bad").  The features are the five most-recent rankings per team prior to last season (which is used to generate the 0/1 target class).  The ranks themselves come from the predicted probability that the team's rank would be "bad" or class 1.  These probabilities were then ranked across all teams with ties resolved by taking the minimum rank.  It's not perfect, but it's good enough to use to predict each team's next season point total.

## Every Team Except Seattle and Vegas

Each team was modeled independently using [Prophet](https://facebook.github.io/prophet/) with yearly seasonality, 1,000 MCMC samples, three lags, and the corresponding season's overall rank as an additional feature.  Holding the most-recent season out of model-training for evaluation, the predicted point total for that season compared to the actual point total has an overall MAPE of 0.0765, which is very good.  The team (model) with the lowest residuals is CGY, of -0.38, while the team with the most residuals is BOS at 20.58.  However, looking at the percentage difference in the residuals, CGY is still the lowest at 0.41%, while CBJ is now the worst, at 24.8% (with BOS fourth-worst at 16.5%).  If it helps, we can do 1 - this percentage difference to determine how confident we can be in my models, so in other words, we can be 75.2% confident in my prediction for CBJ and 99.6% confident in my prediction for CGY, with an average confidence score of 92.3% across all models (teams).

Uncertainty is based on a 95% "uncertainty" interval (Prophet's terminology) which, interestingly enough, isn't equal on either side.  Prophet produces a point prediction along with a "lower" and "upper", and _prediction - lower_ and _upper - prediction_ are not equivalent. :thinking:  They're often close, but strangely, for ANA, they differed by 2.04.  In order to produce a single uncertainty value per team, I took the mean of _prediction - lower_ and _upper - prediction_.

## Seattle

Since Seattle has only played two seasons so far, there's not much we can do with Seattle alone, so instead we look to the score distributions of other teams and the entire NHL, as represented in our data.

Seattle's first season's point total was 60 and second was 100.  We can look at other teams to see which of them had a similar-ish pattern, which winds up being DAL with 60 points in 2021 followed by 98 points in 2022, and also LAK with 59 points in 2013 followed by 100 points in 2014.  After those two seasons, those teams 2023 and 2015 seasons ended with 108 and 95 points, respectively.  We can use this information to fit a normal distribution and consider that our prior.  Our likelihood is what happened so far in Seattle, with which we fit another normal distribution.  If we generate both prior and likelihood PDFs over the total range of points possible in the entire NHL according to our data, and multiply the prior by the likelihood, we get a posterior distribution of possible scores for Seattle for the 2023-2024 season.

However, this isn't quite informed enough, so we're going to repeat this process using this posterior as our prior and the normal distribution of point totals across the entire NHL as our likelihood.  The most likely point total for Seattle in this posterior distribution is 97, with an uncertainty based on the 95th quantile of 15.5.

## Vegas

So far, VGK has only played six seasons, not enough for Prophet to be able to predict anything reasonable.  Instead, I used the [bootstrap method](https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/) and fit [Autogressions](https://www.statsmodels.org/stable/generated/statsmodels.tsa.ar_model.AutoReg.html) with one lag on 250 random resamplings of the six-season time series with replacement.  The predicted point total is the mean of the 250 Autoregression forecasts and the uncertanty is based on a 95% prediction interval calculated with the standard deviation of the 250 forecasts.  The past ranks weren't used since, given their short length, the Autoregression models couldn't figure out what to do with them lol, so they just added noise to the forecasts.

## Limitations

For most teams, the predicted uncertainty is pretty big, in the nature of 20+ points.  This demonstrates a lot of, well, uncertainty in the predictions, despite our confidence in each model's performance (which is based on the point totals).  And this makes sense: there's a lot more to predicting a team's regular-season performance than just their past point totals.  Except maybe for Calgary.

But anyway, this was my first time really sitting down and working on this, this being a problem people devote their entire lives to and I think I can just walk in here... :zany_face:  Well anyway, hopefully this is an interesting benchmark, if nothing else.  I hope to revisit this half-way through the season and see which of these teams are "on track" towards my models' predictions.

Thanks :smile: