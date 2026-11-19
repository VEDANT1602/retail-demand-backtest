# Retail Demand Backtest

Can a gradient-boosted model forecast daily SKU demand more accurately than last week's observation? The benchmark holds out days 150–179, fits features only from data available at forecast time, and selects tree size on a separate validation window. Scheduled promotions are known in advance. [Experiment code](experiment.py)

| Synthetic test population | Boosted WAPE | Seven-day seasonal WAPE |
| --- | ---: | ---: |
| 4,800 store/SKU/day observations | 0.2239 | 0.3999 |
| Cold SKUs | 0.2278 | 0.4215 |

The [WAPE regression](tests/test_experiment.py#L34) recomputes the forecast error from the committed [prediction rows](results/predictions.csv). The [metrics](results/metrics.json) also show how results change under different underforecast penalties. Those penalties are in arbitrary units. Real replenishment decisions would require on-hand inventory, availability and stockout observations.

The [data card](docs/data-card.md) describes the generated demand and new-SKU regime. Numeric store and SKU codes are a modeling convenience. Only load the [model pickle](results/model.pkl) from a trusted source. Main experiment runs use a single native learner thread. Small floating-point differences are possible across platforms.

## Backtest locally

Activate a Python 3.11 virtual environment before installing the versions pinned here.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py --output runs/local
```

[Project reference](https://github.com/shsarv/Machine-Learning-Projects/tree/028b32244cb688cb1beb0ca90754596aeef5e812/TimeSeries%20Multi%20StoreSales%20prediction) · [Source and artifact hashes](results/manifest.json) · [Project provenance](docs/provenance.md)
