# Forecast decision report

The decision is which forecast should feed a replenishment discussion, not whether a notebook can fit sales.

I generated 22688 fictional store/SKU/day observations and kept days 150–179 untouched for evaluation. A seven-day seasonal forecast is a useful control; it can miss planned promotions and new SKU behavior. The fitted gradient booster uses past-only lags and the promotion schedule available before the day begins. Validation selects tree size; test outcomes never tune it.

Test WAPE is 0.2239 for boosting and 0.3999 for seasonal naive across 4800 rows. New-SKU WAPE is 0.2278 versus 0.4215. These error rates belong to the generator, not a retailer.

The cost table in metrics.json varies the penalty for an underforecast. It is a sensitivity analysis in arbitrary units, not stockout savings. The next decision is a representative backtest with real availability and inventory constraints; this experiment alone cannot select a replenishment policy.
