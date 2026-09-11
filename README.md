# Credit Default Risk

A reproducible baseline for predicting loan default from a single application
form, built on the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
dataset.

The goal is not a leaderboard position. It is a pipeline that an underwriting
team could argue with: honest validation, metrics that map to a lending
decision, and an explicit account of where the model fails.

## The problem

A lender sees an application and must decide whether the applicant will default.
Roughly eight percent of applicants do, so a model that predicts "no default"
for everyone is right 92 percent of the time and completely useless. Accuracy is
the wrong instrument.

## Metrics and why

| Metric | What it answers |
| --- | --- |
| ROC AUC | Does the model rank a random defaulter above a random non-defaulter? |
| PR AUC | How much of the precision survives at the low base rate? |
| Gini | The same ranking quality, in the units a risk team already uses |
| KS | Where does the score separate good from bad most sharply? |
| Top-decile default rate | If we decline the riskiest ten percent, what do we actually avoid? |

The last row is the one a business reads. The rest are there to keep the first
one honest.

## Validation

Five-fold stratified cross-validation. Every reported number comes from
out-of-fold predictions, so no row is ever scored by a model that trained on it.
The spread across folds is reported next to the mean, because a mean without a
spread cannot be trusted.

## Getting the data

The competition data needs a Kaggle account and an accepted set of competition
rules, so it is not committed here and cannot be downloaded unattended.

```bash
pip install kaggle
kaggle competitions download -c home-credit-default-risk -p data/raw
unzip data/raw/home-credit-default-risk.zip -d data/raw
```

To run the pipeline without any of that, generate a synthetic stand-in with the
same schema, the same sentinel values and the same missingness:

```bash
python scripts/make_synthetic.py --rows 20000
```

It is a smoke-test fixture. Scores from it mean nothing about credit risk.

## Running

```bash
python -m credit_risk.cli --folds 5 --tag baseline
```

Results are printed and written to `reports/<tag>.json`.

## Layout

```
src/credit_risk/
  config.py      paths, seed, CV settings
  data.py        loading the application table
  features.py    domain ratios and categorical handling
  model.py       LightGBM parameters
  validation.py  out-of-fold cross-validation
  metrics.py     the scoring panel
  cli.py         entry point
scripts/         synthetic data generator
tests/           smoke tests for the pipeline
```

## Notes on the data

`DAYS_EMPLOYED` carries `365243` for applicants who have never been employed.
Left alone, the model happily learns a thousand-year employment history as a
feature. It is replaced with a null in `features.py`. This kind of sentinel is
the most common way a tabular baseline quietly goes wrong.

## Status

Baseline running on synthetic data. Real-data results, error analysis and the
related tables (bureau, previous applications) are next.
