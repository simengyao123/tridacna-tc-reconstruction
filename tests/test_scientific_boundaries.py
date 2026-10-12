"""Regression checks for calendar gaps, TDI padding and event duration boundaries."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from model_core import build_sliding_window
from tdi import add_daily_tdi_features, extract_tdi_features
from extract_events import extract_binary_events, balanced_split_event

class ScientificBoundaries(unittest.TestCase):
    def test_sequences_never_bridge_seasonal_gaps(self):
        x = np.array([[1.],[2.],[100.],[200.]])
        dates = pd.Series(pd.to_datetime(['2013-11-29','2013-11-30','2014-06-01','2014-06-02']))
        seq, labels = build_sliding_window(x,np.array([0,1,0,1]),dates,2,1)
        np.testing.assert_array_equal(seq[:,:,0],[[1,1,1,2],[1,1,2,2],[100,100,100,200],[100,100,200,200]])
        np.testing.assert_array_equal(labels,[0,1,0,1])

    def test_tdi_even_window_orientation_and_edges(self):
        df = pd.DataFrame({'DGR':np.arange(1.,21.),'DGRV':np.arange(20.)/10})
        result = add_daily_tdi_features(df,14)
        expected = extract_tdi_features(df.iloc[3:17])
        np.testing.assert_allclose(result.loc[10,['TDI_A','TDI_V','TDI_F','TDI_E']].to_numpy(dtype=float), expected)
        padded = pd.concat([df.iloc[[0]]]*7+[df.iloc[:7]],ignore_index=True)
        np.testing.assert_allclose(result.loc[0,['TDI_A','TDI_V','TDI_F','TDI_E']].to_numpy(dtype=float),extract_tdi_features(padded))
        self.assertEqual(result.loc[0,'edge_flag'],1)
        self.assertEqual(result.loc[10,'edge_flag'],0)

    def test_events_filter_before_merging_and_do_not_cross_segments(self):
        df = pd.DataFrame({'date':pd.date_range('2000-01-01',periods=9),
                           'segment_id':[1]*7+[2]*2,'flag':[1,1,0,1,0,1,1,1,1]})
        self.assertEqual(extract_binary_events(df,'flag',2,3),[(0,6),(7,8)])
        spans = balanced_split_event(0,19,7)
        self.assertEqual(sum(b-a+1 for a,b in spans),20)
        self.assertTrue(all(b-a+1<=7 for a,b in spans))

if __name__ == '__main__':
    unittest.main()
