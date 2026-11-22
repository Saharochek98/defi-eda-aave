import unittest
import numpy as np
import pandas as pd
from src.stage41 import correlations, normalized, event_order, reserve_quality
from src.metrics import log_returns

class ReviewTests(unittest.TestCase):
    def test_impossible_liquidity_is_flagged_not_clipped(self):
        frame=pd.DataFrame({'available_token':[-1.,1.],'utilization_reported':[1.01,.9],'supply_token':[100.,100.],'borrow_token':[90.,90.]})
        out=reserve_quality(frame)
        self.assertTrue(out.quality_invalid.iloc[0])
        self.assertTrue(pd.isna(out.utilization_valid.iloc[0]))
        self.assertEqual(out.utilization_reported.iloc[0],1.01)
        self.assertEqual(out.utilization_valid.iloc[1],.9)
    def test_numeric_event_order_not_hash_or_lexical_order(self):
        rows=[{'id':'9:2:0xffff:100:4'},{'id':'10:1:0x0000:9:1'},{'id':'10:1:0x0000:10:2'}]
        self.assertEqual(max(rows,key=event_order),rows[-1])
        with self.assertRaises(ValueError):event_order({'id':'unknown'})

    def test_spearman_uses_average_tied_ranks(self):
        out=correlations(pd.Series([1.,1.,2.,3.]),pd.Series([1.,2.,3.,4.]))
        self.assertAlmostEqual(out['spearman'],.9486832980505138)

    def test_pairwise_alignment_uses_dates_not_position(self):
        a=pd.Series([1.,2.,3.,4.],index=[0,1,2,3])
        b=pd.Series([20.,30.,40.,500.],index=[1,2,3,4])
        out=correlations(a,b)
        self.assertEqual(out['n'],3)
        self.assertAlmostEqual(out['pearson'],1)

    def test_rank_and_trim_are_sensitive_to_a_single_extreme(self):
        a=pd.Series(np.arange(100,dtype=float));b=a.copy();b.iloc[-1]=-10000
        out=correlations(a,b)
        self.assertGreater(out['spearman'],.9)
        self.assertAlmostEqual(out['trim_pearson'],1)
        self.assertLess(out['pearson'],0)
        self.assertEqual(b.iloc[-1],-10000)

    def test_common_base_not_individual_first_valid_dates(self):
        a=pd.DataFrame({'a':[20.,40.,np.nan],'b':[10.,15.,20.]})
        out=normalized(a)
        self.assertTrue(out.iloc[0].eq(100).all())
        self.assertEqual(out.a.iloc[1],200)
        self.assertTrue(pd.isna(out.a.iloc[2]))
        a.iloc[0,0]=np.nan
        with self.assertRaises(ValueError):normalized(a)

    def test_flow_change_is_shifted_to_interval_end(self):
        dates=pd.date_range('2026-01-01',periods=4,tz='UTC')
        flow=pd.Series([10.,20.,80.,80.],index=dates)
        shifted=log_returns(flow).shift(1)
        self.assertTrue(shifted.iloc[:2].isna().all())
        self.assertAlmostEqual(shifted.iloc[2],np.log(2))
        self.assertAlmostEqual(shifted.iloc[3],np.log(4))

if __name__=='__main__':unittest.main()
