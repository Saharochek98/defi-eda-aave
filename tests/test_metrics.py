"""Numerical unit checks; test fixtures never enter the research datasets."""

import math
import unittest

import numpy as np
import pandas as pd

from src.metrics import safe_ratio, log_returns, ray_to_apy, annualized_volatility, lending_snapshot, RESERVES
from src.onchain import POOL


class MetricsTests(unittest.TestCase):
    def test_zero_or_missing_denominator_stays_unknown(self):
        result=safe_ratio(pd.Series([10.,10.,10.]),pd.Series([0.,np.nan,2.]))
        self.assertTrue(result.iloc[:2].isna().all())
        self.assertEqual(result.iloc[2],5.)

    def test_log_returns_never_bridge_missing_day(self):
        dates=pd.date_range('2026-01-01',periods=4,tz='UTC')
        result=log_returns(pd.Series([100.,np.nan,121.,133.1],index=dates))
        self.assertTrue(result.iloc[:3].isna().all())
        self.assertAlmostEqual(result.iloc[3],math.log(1.1))

    def test_nonpositive_price_does_not_produce_infinite_return(self):
        result=log_returns(pd.Series([100.,0.,110.]))
        self.assertTrue(result.isna().all())

    def test_volatility_uses_crypto_calendar_and_sample_variance(self):
        self.assertAlmostEqual(annualized_volatility(pd.Series([-.01,.01])),.01*math.sqrt(2*365))
        with self.assertRaises(ValueError):
            annualized_volatility(pd.Series([-.01,.01]),0)

    def test_ray_scaling_and_compounding(self):
        self.assertEqual(ray_to_apy('0'),0)
        self.assertAlmostEqual(ray_to_apy(str(5*10**25)),math.expm1(.05),places=9)
        with self.assertRaises(ValueError):
            ray_to_apy('-1')

    def fixtures(self):
        return [{'underlyingAsset':asset,'pool':{'pool':POOL},'decimals':decimals,
                 'lastUpdateTimestamp':1793491187,'totalATokenSupply':str(200*10**decimals),
                 'totalCurrentVariableDebt':str(100*10**decimals),'totalPrincipalStableDebt':'0',
                 'availableLiquidity':str(100*10**decimals),'utilizationRate':'.5',
                 'liquidityRate':'0','variableBorrowRate':'0'} for asset,decimals in RESERVES.values()]

    def test_reserve_decimals_keep_native_units(self):
        result=lending_snapshot(self.fixtures())
        self.assertTrue(result.supply_token.eq(200.).all())
        self.assertTrue(result.borrow_token.eq(100.).all())

    def test_stable_principal_is_not_treated_as_current_stable_debt(self):
        rows=self.fixtures();rows[0]['totalPrincipalStableDebt']='1'
        result=lending_snapshot(rows)
        self.assertTrue(pd.isna(result.borrow_token.iloc[0]))
        self.assertEqual(result.variable_borrow_token.iloc[0],100.)

    def test_wrong_decimals_are_rejected(self):
        rows=self.fixtures();rows[0]['decimals']=18
        with self.assertRaisesRegex(ValueError,'Unexpected decimals'):
            lending_snapshot(rows)


if __name__=='__main__':
    unittest.main()
