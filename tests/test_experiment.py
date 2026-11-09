import csv,json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiment import features,generate,arrays,wape,loss
from experiment_io import verify_manifest

class ForecastContracts(unittest.TestCase):
    def test_known_wape_and_cost(self):
        self.assertAlmostEqual(wape([2,4],[1,5]),2/6)
        self.assertAlmostEqual(loss(np.array([2,4]),np.array([1,5]),3),2)
    def test_zero_denominator_is_not_fake_zero(self):
        self.assertIsNone(wape([0,0],[1,1]))
    def test_feature_builder_rejects_future_history(self):
        with self.assertRaises(ValueError):features([1,2,3],0,0,2,0)
    def test_time_and_entity_codes_cannot_bypass_with_nan_or_bool(self):
        for value in [float('nan'),float('inf'),True,1.5,-1]:
            for index in range(3):
                args=[0,0,5];args[index]=value
                with self.subTest(value=value,index=index),self.assertRaises(ValueError):features([],args[0],args[1],args[2],0)
    def test_lags_use_only_supplied_history(self):
        x=features(list(range(1,9)),0,0,8,1)
        self.assertEqual(x[5],8);self.assertEqual(x[6],2);self.assertEqual(x[7],5)
    def test_malformed_history_rejected(self):
        for history in [[float('nan')],[-1]]:
            with self.assertRaises(ValueError):features(history,0,0,5,0)
    def test_temporal_and_cold_boundaries(self):
        rows=generate();_,_,day=arrays(rows)
        self.assertLess(day[day<120].max(),day[day>=120].min())
        self.assertTrue(all(r['day']>=135 for r in rows if r['cold_sku']))
        with (ROOT/'results/data.csv').open(encoding='utf-8',newline='')as file:recorded=list(csv.DictReader(file))
        self.assertEqual(len(rows),len(recorded));self.assertEqual(int(recorded[100]['units']),rows[100]['units'])
    def test_integrity(self):verify_manifest()
    def test_raw_forecast_error_replays_reported_metric(self):
        with (ROOT/'results/predictions.csv').open(encoding='utf-8',newline='')as file:r=list(csv.DictReader(file))
        m=json.loads((ROOT/'results/metrics.json').read_text())
        y=np.array([float(x['actual'])for x in r]);p=np.array([float(x['boosted'])for x in r])
        self.assertAlmostEqual(wape(y,p),m['test']['boosted_wape'])

if __name__=='__main__':unittest.main()
