"""Synthetic invariants for motion features, never reported as recognition scores."""
from dataclasses import replace
import unittest

import numpy as np
import pandas as pd

import group_model_study as base
import imu_motion_features as m
import sensor_preprocessing as prep


def captures(shift=0):
    values=np.zeros((80,12))
    values[:,8]=1.
    values[:,0]=250.
    values[20+shift:40+shift,9]=40.
    values[20+shift:40+shift,6]=np.sin(np.linspace(0,np.pi,20))
    return {h:prep.HandCapture('a','subject1','s','t','gesture','training',h,'XIAO','v1',
        'capture.csv','0'*64,'2026-11-18T00:00:00Z','2026-11-18T00:00:08Z',False,
        np.arange(80)*100.+(0 if h=='left' else 1234567),values.copy()) for h in ['left','right']}


class MagnitudeTests(unittest.TestCase):
    def test_norms_invariant_under_rigid_axis_rotation(self):
        rng=np.random.default_rng(5)
        values=rng.normal(size=(50,12))
        q,_=np.linalg.qr(rng.normal(size=(3,3)))
        rotated=values.copy()
        rotated[:,6:9]=values[:,6:9]@q
        rotated[:,9:12]=values[:,9:12]@q
        np.testing.assert_allclose(m.magnitudes(values),m.magnitudes(rotated),rtol=1e-12,atol=1e-12)

    def test_original_schema_and_values_unchanged(self):
        hands=captures()
        result,_=m.features(hands,prep.Recipe())
        self.assertEqual(result['original'],base.trial_features(hands,prep.Recipe()))
        self.assertEqual([len(result[k]) for k in m.VARIANTS],[384,448,388,452])

    def test_bad_sensor_shape_and_nonfinite_values_rejected(self):
        with self.assertRaises(ValueError):
            m.magnitudes(np.zeros((3,11)))
        with self.assertRaises(ValueError):
            m.magnitudes(np.full((5,12),np.nan))


class AlignmentTests(unittest.TestCase):
    def test_shifted_motion_aligns_without_changing_whole_statistics(self):
        a,_=m.features(captures(),prep.Recipe())
        b,_=m.features(captures(15),prep.Recipe())
        for key in a['magnitude_aligned']:
            self.assertAlmostEqual(a['magnitude_aligned'][key],b['magnitude_aligned'][key],places=10)
        self.assertNotEqual(a['original']['phase0_left_gx_mean'],b['original']['phase0_left_gx_mean'])

    def test_crop_resamples_only_within_valid_bounds(self):
        hand=captures()['left']
        values,times,log=m.motion_window(hand.values,hand.times_ms)
        self.assertEqual(log['status'],'aligned')
        self.assertEqual(values.shape,(32,12))
        self.assertGreaterEqual(times[0],hand.times_ms[0])
        self.assertLessEqual(times[-1],hand.times_ms[-1])
        self.assertGreater(log['start_index'],0)
        self.assertLess(log['end_index'],len(hand.values)-1)
        self.assertTrue((np.diff(times)>0).all())

    def test_still_hand_preserves_pose_and_fsr_with_full_window_fallback(self):
        hand=captures()['left']
        original=hand.values.copy(); original[:,9:12]=0
        values,times,log=m.motion_window(original,hand.times_ms)
        self.assertEqual(log['status'],'no_sustained_activity')
        np.testing.assert_array_equal(values,original)
        np.testing.assert_array_equal(times,hand.times_ms)
        self.assertTrue((values[:,8]==1).all())
        self.assertTrue((values[:,0]==250).all())

    def test_isolated_spike_never_becomes_a_motion_window(self):
        hand=captures()['left']
        values=hand.values.copy(); values[:,9:12]=0; values[10,9]=200
        _,_,log=m.motion_window(values,hand.times_ms)
        self.assertFalse(log['resampled'])

    def test_large_internal_gap_falls_back_without_interpolation(self):
        hand=captures()['left']
        times=hand.times_ms.copy(); times[30:]+=300
        result,returned,log=m.motion_window(hand.values,times)
        self.assertEqual(log['status'],'resample_gap_too_large')
        np.testing.assert_array_equal(returned,times)
        np.testing.assert_array_equal(result,hand.values)

    def test_independent_clock_origins_and_inputs_unchanged(self):
        hands=captures()
        before={h:c.values.copy() for h,c in hands.items()}
        result,_=m.features(hands,prep.Recipe(denoise=True))
        hands['right']=replace(hands['right'],times_ms=hands['right'].times_ms+980000)
        again,_=m.features(hands,prep.Recipe(denoise=True))
        for variant in result:
            np.testing.assert_allclose(list(result[variant].values()),list(again[variant].values()),atol=1e-10)
        for h in hands:
            np.testing.assert_array_equal(hands[h].values,before[h])

    def test_missing_hand_and_invalid_policy_rejected(self):
        with self.assertRaisesRegex(ValueError,'Both hands'):
            m.features({'left':captures()['left']},prep.Recipe())
        with self.assertRaises(ValueError):
            m.MotionPolicy(resample_points=3)
        with self.assertRaises(ValueError):
            m.MotionPolicy(pad_ms=-1)

    def test_sensor_selection_keeps_norms_and_excludes_pressure_from_imu(self):
        result,_=m.features(captures(),prep.Recipe())
        tables={('raw',v):pd.DataFrame([rows]) for v,rows in result.items()}
        X,columns=m.matrices(tables)
        self.assertEqual(X['raw__magnitude_aligned__imu'].shape,(1,260))
        cols=columns['raw__magnitude_aligned__imu']
        self.assertTrue(any('accel_norm' in c for c in cols))
        self.assertTrue(all('_fsr' not in c for c in cols))


if __name__=='__main__':
    unittest.main()
