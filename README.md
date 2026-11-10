This repository is focused on RUL estimation for a high-speed CNC milling machine (PHM data challenge 2010).

## Setup

```bash
just dev-sync
cp .env.example .env   # paths to the config, data, checkpoints and reports
```

Put the PHM 2010 cutters in `data/` (`data/c1/c1/c_1_001.csv`, `data/c1/c1_wear.csv`, ...).

## Run

Hyperparameters come from `config/config.yaml`.

```bash
uv run main.py --mode train --cutter c4   # train on c1 + c6, test on c4
uv run main.py --mode train-all           # leave-one-cutter-out over c1, c4, c6
uv run main.py --mode tune --trials 20    # hyperparameter search
```

Each test cutter gets `reports/<output_artifacts>/<cutter>/predicted_vs_measured.png`.

## Hyperparameter tuning

Random search over `SEARCH_SPACE` in `src/tool_wear_phm2010/ml_pipeline/tuning.py`. Trying the full grid is infeasible, so it draws `--trials` random configs.

It's nested, so the test cutter never helps choose the hyperparameters. For each test cutter:

1. Score every config on the other two cutters: train on one, test on the other, then swap.
2. Retrain the best config on both and test once on the held-out cutter.

Results go to `reports/tuning/results.csv` (updated after every trial) and `reports/tuned/<cutter>/`. Tuned values are not written back to `config.yaml`; copy them there by hand.
