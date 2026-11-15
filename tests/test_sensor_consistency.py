"""Synthetic checks for descriptive waveform comparisons; not measured scores."""
import unittest
import numpy as np
import pandas as pd
from tools import compare_sign_sensor_consistency as c


class SensorConsistencyTests(unittest.TestCase):
    def test_identical_people_have_zero_distance_with_distinct_signs(self):
        low = np.zeros((2,50,12))
        high = np.ones((2,50,12))
        rows,pairs,_ = c.compare_templates({('a','subject1'):low,('a','subject2'):low,
                                           ('b','subject1'):high,('b','subject2'):high})
        for r in rows:
            self.assertEqual(r['FSR_person_distance'],0)
            self.assertEqual(r['IMU_difference_ratio'],0)
            self.assertEqual(r['FSR_dynamic_pairs'],0)
        self.assertEqual(len(pairs),4)

    def test_flat_values_are_not_awarded_perfect_motion_similarity(self):
        self.assertIsNone(c.shape_cosine(np.zeros((2,50,6)),np.zeros((2,50,6))))
        ramp = np.broadcast_to(np.linspace(0,1,50)[None,:,None],(2,50,6)).copy()
        self.assertAlmostEqual(c.shape_cosine(ramp,ramp),1)
        self.assertAlmostEqual(c.shape_cosine(ramp,-ramp),-1)
        self.assertGreater(c.rms_distance(ramp,ramp+4),0)
        self.assertAlmostEqual(c.shape_cosine(ramp,ramp+4),1)

    def test_channel_unit_changes_do_not_change_scaled_distances(self):
        rng=np.random.default_rng(7)
        templates={(s,p):rng.normal(size=(2,50,12)) for s in ['a','b'] for p in ['subject1','subject2']}
        factors=np.arange(1,13)[None,None,:]*100
        original,_,_=c.compare_templates(templates)
        scaled,_,_=c.compare_templates({k:v*factors for k,v in templates.items()})
        for a,b in zip(original,scaled):
            self.assertAlmostEqual(a['FSR_person_distance'],b['FSR_person_distance'])
            self.assertAlmostEqual(a['IMU_difference_ratio'],b['IMU_difference_ratio'])

    def test_waveform_relative_clock_and_raw_zero_flags(self):
        from test_six_sign_diagnostics import raw_trial
        frame=raw_trial()
        fsr=[f'fsr{i}' for i in range(1,7)]
        frame.loc[frame.hand.eq('left'),fsr]=0
        before=frame.copy(deep=True)
        wave,flags=c.waveform(frame)
        self.assertEqual(wave.shape,(2,50,12))
        self.assertTrue(flags['left_fsr_all_zero'])
        self.assertFalse(flags['right_fsr_all_zero'])
        shifted=frame.copy()
        shifted.loc[shifted.hand.eq('right'),'time_ms']+=700000
        np.testing.assert_allclose(c.waveform(shifted)[0],wave)
        pd.testing.assert_frame_equal(frame,before)
        with self.assertRaisesRegex(ValueError,'Missing hand'):
            c.waveform(frame[frame.hand.eq('left')])


if __name__ == '__main__':
    unittest.main()
