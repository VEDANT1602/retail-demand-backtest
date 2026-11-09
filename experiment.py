"""Forecast a fictional retail panel using only history and known promotion schedules."""
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from experiment_io import csv_write, dump, model_write, freeze

FEATURES = ["store", "sku", "weekday_sin", "weekday_cos", "planned_promotion", "lag_1", "lag_7", "mean_prior_7", "history_count"]


def features(history, store, sku, day, promo):
    """History is explicitly truncated by the caller at the prediction instant."""
    for value,name in [(store,'store'),(sku,'sku'),(day,'day')]:
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<0:
            raise ValueError(name+' must be a nonnegative integer')
    h = np.asarray(history, dtype=float)
    if h.ndim != 1 or not np.isfinite(h).all() or (h < 0).any():
        raise ValueError("history must be finite nonnegative observations")
    if day < len(h) or not np.isfinite(promo) or promo not in (0, 1):
        raise ValueError("future history or invalid planned promotion")
    return [store, sku, np.sin(2*np.pi*day/7), np.cos(2*np.pi*day/7), promo,
            h[-1] if len(h) else 0, h[-7] if len(h)>=7 else 0,
            h[-7:].mean() if len(h) else 0, min(len(h), 7)]


def generate(seed=1602):
    rng = np.random.default_rng(seed)
    rows = []
    for store in range(8):
        for sku in range(20):
            start = 135 if sku >= 16 else 0
            history = []
            base = rng.uniform(5, 22) * (0.8 + store/15)
            for day in range(start, 180):
                promo = int((day + sku*3 + store) % 19 < 3)
                rate = base * (1 + .30*np.sin(2*np.pi*day/7)) * (1 + .8*promo)
                # A declared late-period demand shift tests the stationary learner.
                rate *= 1.25 if day >= 165 and sku % 5 == 0 else 1.0
                units = int(rng.poisson(rate))
                x = features(history, store, sku, day, promo)
                if day >= 14:
                    rows.append(dict(day=day, store=store, sku=sku,
                                     **dict(zip(FEATURES[2:], x[2:])), units=units,
                                     cold_sku=int(start > 0)))
                history.append(units)
    return rows


def arrays(rows):
    x = np.array([[r[k] for k in FEATURES] for r in rows])
    y = np.array([r["units"] for r in rows], dtype=float)
    days = np.array([r["day"] for r in rows])
    return x,y,days


def wape(y,p):
    y,p=np.asarray(y),np.asarray(p)
    if len(y)==0 or y.sum()<=0:
        return None
    return float(np.abs(y-p).sum()/y.sum())


def loss(y,p,under=3):
    return float(np.mean(under*np.maximum(y-p,0)+np.maximum(p-y,0)))


def run(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    rows=generate(); x,y,day=arrays(rows)
    train=day<120; val=(day>=120)&(day<150); test=day>=150
    candidates={}
    for leaves in [7,15]:
        model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=leaves,
            learning_rate=.08,l2_regularization=3,random_state=1602).fit(x[train],y[train])
        candidates[leaves]=(wape(y[val],np.maximum(model.predict(x[val]),0)),model)
    chosen=min(candidates,key=lambda k:candidates[k][0]); model=candidates[chosen][1]
    pred=np.maximum(model.predict(x[test]),0)
    train_mean=float(y[train].mean())
    baseline=np.array([r["lag_7"] if r["history_count"]>=7 else train_mean for r,m in zip(rows,test) if m])
    truth=y[test]; held=[r for r,m in zip(rows,test) if m]
    cold=np.array([r["cold_sku"]==1 for r in held])
    predictions=[dict(day=r["day"],store=r["store"],sku=r["sku"],cold_sku=r["cold_sku"],actual=float(a),seasonal_naive=float(b),boosted=float(p)) for r,a,b,p in zip(held,truth,baseline,pred)]
    scale=np.mean(np.abs(y[train]-x[train,FEATURES.index("lag_7")]))
    metrics={"seed":1602,"rows":len(rows),"train_rows":int(train.sum()),"validation_rows":int(val.sum()),"test_rows":int(test.sum()),
             "split_days":{"train":[14,119],"validation":[120,149],"test":[150,179]},"selected_max_leaves":chosen,
             "validation_wape":{str(k):v[0] for k,v in candidates.items()},
             "test":{"boosted_wape":wape(truth,pred),"seasonal_naive_wape":wape(truth,baseline),
                     "boosted_mase":float(np.abs(truth-pred).mean()/scale),"seasonal_naive_mase":float(np.abs(truth-baseline).mean()/scale),
                     "cold_rows":int(cold.sum()),"cold_boosted_wape":wape(truth[cold],pred[cold]),"cold_naive_wape":wape(truth[cold],baseline[cold])},
             "illustrative_cost":{str(u):{"boosted":loss(truth,pred,u),"naive":loss(truth,baseline,u)} for u in [1,3,8]},
             "mase_scale":"training-only mean absolute seven-day seasonal difference; zero-history rows included",
             "limits":"Original stationary synthetic demand with one declared late shift. Promotions are known schedule inputs. No stockouts, fulfillment or financial ROI inferred."}
    csv_write(output/"data.csv",rows); csv_write(output/"predictions.csv",predictions)
    model_write(output/"model.pkl",model);dump(output/"metrics.json",metrics)
    text=f"""# Forecast decision report

The decision is which forecast should feed a replenishment discussion, not whether a notebook can fit sales.

I generated {len(rows)} fictional store/SKU/day observations and kept days 150–179 untouched for evaluation. A seven-day seasonal forecast is a useful control; it can miss planned promotions and new SKU behavior. The fitted gradient booster uses past-only lags and the promotion schedule available before the day begins. Validation selects tree size; test outcomes never tune it.

Test WAPE is {metrics['test']['boosted_wape']:.4f} for boosting and {metrics['test']['seasonal_naive_wape']:.4f} for seasonal naive across {int(test.sum())} rows. New-SKU WAPE is {metrics['test']['cold_boosted_wape']:.4f} versus {metrics['test']['cold_naive_wape']:.4f}. These error rates belong to the generator, not a retailer.

The cost table in metrics.json varies the penalty for an underforecast. It is a sensitivity analysis in arbitrary units, not stockout savings. The next decision is a representative backtest with real availability and inventory constraints; this experiment alone cannot select a replenishment policy.
"""
    (output/"report.md").write_bytes(text.encode());freeze(output)
    print(json.dumps(metrics["test"],indent=2))
