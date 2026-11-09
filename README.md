# Retail Demand Backtest

**A forecast becomes useful only after its time boundary and next decision are explicit.**

This is original independent portfolio work by Vedant Thaker. It contains runnable code, original fixtures, trained models where applicable, raw decisions/predictions and fresh measured results. All checked-in experiments run locally on a CPU without paid services, API keys or model downloads.

## The decision and implementation

Fit real gradient-boosted demand models on past days, select complexity on a separate validation period, and compare with a seven-day seasonal naive forecast on later observations. Promotions are known schedule inputs. New SKUs and a declared late demand shift remain visible as limits.

## Recorded result

On 4,800 test store/SKU/day observations, boosted WAPE is **0.2239** versus **0.3999** for seasonal naive. Cold-SKU WAPE is 0.2278 versus 0.4215.

Read the [full experiment narrative](results/report.md), [raw metrics](results/metrics.json), [data card](docs/data-card.md), and [source/fixture commitments](results/manifest.json). Model-based projects retain the fitted artifact as `results/model.pkl`; only artifacts generated here should be loaded. Never load an untrusted pickle.

## Reproduce

Python 3.11. From this repository's root:

```bash
python -m venv .venv
# Activate the environment for your shell, then:
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py --output runs/local
```

The alternate output directory preserves the committed reference run. Native learner runs set `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`. Source, input, model and raw-output SHA-256 hashes are committed in the reference manifest. Floating-point values may vary slightly between platforms.

## Limits and next decision

The synthetic demand equation is simple. Numeric SKU/store codes are a modeling convenience, not evidence of meaningful category order. The experiment contains no real stockouts, inventory allocation, replenishment deployment or retailer ROI. MASE uses training-only seven-day absolute error, including zero-history rows. The cost table is arbitrary-weight sensitivity analysis.

Bring a representative availability-aware panel, confirm forecast horizons, and evaluate inventory decisions separately from forecast error.

## Reference and authorship

Conceptual source: [shsarv/Machine-Learning-Projects project entry](https://github.com/shsarv/Machine-Learning-Projects/tree/028b32244cb688cb1beb0ca90754596aeef5e812/TimeSeries%20Multi%20StoreSales%20prediction). The problem/method theme was redesigned independently. No source code, data, weights, assets or benchmark figures were copied. [Provenance](docs/provenance.md) records the pinned source, uncertain source-license boundary and AI implementation assistance. This project demonstrates present methods; it does not establish historical employment or customer outcomes.
