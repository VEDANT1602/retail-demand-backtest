# Data card

## Original source and population

Eight fictional stores and twenty SKUs, with the final four introduced after the training period. Days 14–119 train, 120–149 validate, 150–179 test. Targets are uncensored simulated demand. The model uses historical units, weekday, store/SKU codes and a prepublished promotion flag; no future target enters a feature. The same actual prior sales are available to both methods at each rolling prediction instant. The evaluation is one-step-ahead with realized history, not a thirty-day recursive forecast.

The generator is `generate()` in `experiment.py`, or the authored dialog fixture inside `run()` for reservations. The reference input is in `results/data.csv` or `results/data.json`. No employer, client or personal records are present. All names/IDs are fictional. Original fixtures are MIT licensed with this repository.

## What the data can establish

The synthetic demand equation is simple. Numeric SKU/store codes are a modeling convenience, not evidence of meaningful category order. The experiment contains no real stockouts, inventory allocation, replenishment deployment or retailer ROI. MASE uses training-only seven-day absolute error, including zero-history rows. The cost table is arbitrary-weight sensitivity analysis.

The repository reports actual executed CPU comparisons with their own denominators. Synthetic labels encode a known task; they are not evidence of real-world validity. Data, raw outputs, trained artifacts where present and core source bytes are committed by the SHA-256 manifest. Updating code requires regenerating the experiment before refreshing those commitments.
