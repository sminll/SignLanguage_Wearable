"""SYNTHETIC fixtures only. No real recognition performance is measured here."""
import tempfile
import unittest
import contextlib
import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from evaluate_sign_sequences import FEATURES, MANIFEST_COLUMNS, evaluate, load_model, person, main
from train_sign_knn import SENSORS


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        from sklearn.pipeline import Pipeline
        model = Pipeline([('imputer',SimpleImputer()),('scaler',StandardScaler()),
                          ('knn',KNeighborsClassifier(n_neighbors=1))])
        train = pd.DataFrame([[0 if c.endswith(('mean','min','max')) else 0 for c in FEATURES],
                              [10 if c.endswith(('mean','min','max')) else 0 for c in FEATURES]], columns=FEATURES)
        model.fit(train, ['머리','아프다'])
        self.model = self.root/'knn_model.joblib'
        joblib.dump(dict(pipeline=model,columns=FEATURES,classes=['머리','아프다'],
                         training_sessions=['train_session']), self.model)
        pd.DataFrame([dict(subject_id='subject1',session_id='train_session',hand='left')]).to_csv(
            self.root/'trial_features.csv',index=False)
        self.rows = []
        self.counter = 0

    def attempt(self, outcomes=(True, True), participant='subject2', status='complete'):
        self.counter += 1
        attempt = f'A{self.counter}'
        for step, (label, correct) in enumerate(zip(['머리','아프다'], outcomes),1):
            sid = f'session_{attempt}_{step}'
            value = (0 if step==1 else 10) if correct else (10 if step==1 else 0)
            samples = []
            for seq,t in enumerate(range(0,5000,100),1):
                samples.append(dict(subject_id=participant,session_id=sid,trial_id=label+'1',hand='left',
                                    label=label,source='REAL',time_ms=90000+t,sample_seq=seq,
                                    host_elapsed_ms=t,trial_elapsed_ms=t,imu_ok=1,
                                    **{s:value for s in SENSORS}))
            path=self.root/f'{sid}.csv'
            pd.DataFrame(samples).to_csv(path,index=False)
            start=(self.counter-1)*16000+(step-1)*8000
            self.rows.append(dict(participant_id=participant,attempt_id=attempt,step=str(step),actual_label=label,
                                  csv_path=path.name,session_id=sid,trial_id=label+'1',hand='left',
                                  recording_id='one_unpaused_clock',window_start_ms=start,
                                  window_end_ms=start+5000,status=status))

    def run_eval(self, **kwargs):
        manifest=self.root/'manifest.csv'
        pd.DataFrame(self.rows,columns=MANIFEST_COLUMNS).to_csv(manifest,index=False)
        return evaluate(manifest,self.model,self.root/'output',data_kind='synthetic',**kwargs)

    def change_raw(self, row=0, fn=None):
        path=self.root/self.rows[row]['csv_path']
        frame=pd.read_csv(path)
        fn(frame).to_csv(path,index=False)

    def test_empirical_joint_is_not_product_of_marginals(self):
        self.attempt(); self.attempt(); self.attempt((False,False))
        result=self.run_eval(mode='new_participant')
        self.assertEqual(result['valid_attempts'],3)
        self.assertEqual(result['successes'],2)
        self.assertEqual(result['failures'],1)
        self.assertAlmostEqual(result['combination_accuracy_pct'],200/3)
        self.assertNotAlmostEqual(result['combination_accuracy_pct'],(2/3)**2*100)
        frame=pd.read_csv(self.root/'output/attempt_results.csv')
        self.assertEqual(frame.predicted_sentence.iloc[0],'머리가 아픕니다')
        self.assertEqual(result['status'],'테스트용 가상 데이터')
        for f in ['summary.csv','summary.json','report.md','combination_accuracy.png','participant_summary.csv','manifest_snapshot.csv']:
            self.assertTrue((self.root/'output'/f).is_file())

    def test_missing_step_excluded_from_all_denominators(self):
        self.attempt(); self.attempt(); self.rows.pop()
        result=self.run_eval()
        self.assertEqual((result['total_attempts'],result['valid_attempts'],result['excluded_attempts']),(2,1,1))
        self.assertEqual(result['first_sign_accuracy_pct'],100)
        self.assertEqual(result['exclusion_reasons'],{'missing_or_duplicate_step':1})

    def test_one_wrong_is_failure_not_exclusion(self):
        self.attempt((True,False))
        result=self.run_eval()
        self.assertEqual((result['valid_attempts'],result['failures'],result['successes']),(1,1,0))
        self.assertEqual(result['second_sign_accuracy_pct'],0)

    def test_empty_manifest_is_unmeasured(self):
        result=self.run_eval()
        self.assertEqual(result['status'],'미측정')
        self.assertIsNone(result['combination_accuracy_pct'])

    def test_declared_synthetic_cannot_be_reported_as_real(self):
        self.attempt()
        manifest=self.root/'manifest.csv'
        pd.DataFrame(self.rows).assign(data_kind='synthetic').to_csv(manifest,index=False)
        with self.assertRaisesRegex(ValueError,'data_kind'):
            evaluate(manifest,self.model,self.root/'output',data_kind='real')
        self.assertFalse((self.root/'output').exists())

    def test_incomplete_status_and_all_rejected_is_unmeasured(self):
        self.attempt(status='interrupted')
        result=self.run_eval()
        self.assertEqual(result['excluded_attempts'],1)
        self.assertIsNone(result['first_sign_accuracy_pct'])

    def test_short_clock_window_rejected(self):
        self.attempt()
        self.change_raw(fn=lambda f:f.iloc[:20])
        self.assertIn('incomplete_5s_window',self.run_eval()['exclusion_reasons'])

    def test_internal_gap_rejected(self):
        self.attempt()
        self.change_raw(fn=lambda f:f.loc[~f.index.isin(range(20,30))])
        self.assertIn('sensor_gap_over_500ms',self.run_eval()['exclusion_reasons'])

    def test_missing_imu_rejected_not_imputed(self):
        self.attempt()
        def missing(f):
            f.loc[2,'ax']=np.nan
            return f
        self.change_raw(fn=missing)
        self.assertIn('missing_or_nonfinite_sensor_values',self.run_eval()['exclusion_reasons'])

    def test_duplicate_packet_rejected(self):
        self.attempt()
        self.change_raw(fn=lambda f:pd.concat([f,f.iloc[:1]],ignore_index=True))
        self.assertIn('duplicate_sample_sequence',self.run_eval()['exclusion_reasons'])

    def test_incorrect_truth_metadata_rejected(self):
        self.attempt()
        self.change_raw(fn=lambda f:f.assign(label='갑자기'))
        self.assertIn('raw_label_mismatch',self.run_eval()['exclusion_reasons'])

    def test_reversed_steps_schedule_rejected(self):
        self.attempt()
        self.rows[0]['window_start_ms']=16000
        self.rows[0]['window_end_ms']=21000
        self.assertIn('not_5s_rest3s_5s_schedule',self.run_eval()['exclusion_reasons'])

    def test_different_clock_rejected(self):
        self.attempt();self.rows[1]['recording_id']='other_clock'
        self.assertIn('different_recording_clocks',self.run_eval()['exclusion_reasons'])

    def test_reused_window_rejected(self):
        self.attempt()
        copies=[dict(row,attempt_id='copy') for row in self.rows]
        self.rows.extend(copies)
        self.assertEqual(self.run_eval()['excluded_attempts'],2)

    def test_training_session_overlap_fails_closed(self):
        self.attempt();self.rows[0]['session_id']='train_session'
        with self.assertRaisesRegex(ValueError,'overlap'):
            self.run_eval(mode='new_participant')
        self.assertFalse((self.root/'output').exists())

    def test_subject_alias_cannot_pass_as_unseen_person(self):
        self.attempt(participant='subject-1')
        result=self.run_eval(mode='new_participant')
        self.assertIn('participant_was_used_in_training',result['exclusion_reasons'])
        self.assertEqual(person('subject-8'),person('subject8'))

    def test_same_person_new_session_separate_mode(self):
        self.attempt(participant='subject1')
        result=self.run_eval(mode='same_participant_new_session')
        self.assertEqual(result['valid_attempts'],1)

    def test_unknown_training_provenance_cannot_claim_independence(self):
        (self.root/'trial_features.csv').unlink()
        self.attempt()
        with self.assertRaisesRegex(ValueError,'provenance'):
            self.run_eval(mode='new_participant')

    def test_unsupported_signs_and_wrong_feature_model_rejected(self):
        with self.assertRaisesRegex(ValueError,'does not support'):
            load_model(self.model,('교통사고','상처'))
        bundle=joblib.load(self.model);bundle['columns']=['ax_mean']
        joblib.dump(bundle,self.model)
        with self.assertRaisesRegex(ValueError,'48-feature'):
            load_model(self.model,('머리','아프다'))

    def test_output_and_input_preserved(self):
        self.attempt()
        before={p:p.read_bytes() for p in self.root.iterdir() if p.is_file()}
        self.run_eval()
        for p,data in before.items():
            self.assertEqual(p.read_bytes(),data)
        with self.assertRaisesRegex(ValueError,'Output exists'):
            self.run_eval()

    def test_matlab_local_csv_schema(self):
        self.attempt()
        for row in self.rows:
            path=self.root/row['csv_path']; f=pd.read_csv(path)
            f=f.drop(columns=['source','hand','label'])
            f=f.rename(columns={'trial_id':'source','sample_seq':'seq',
                                **{f'fsr{i}':f'fsr{i}_raw' for i in range(1,7)},
                                **dict(zip(SENSORS[6:],['ax_g','ay_g','az_g','gx_dps','gy_dps','gz_dps']))})
            f['device_id']='L';f.to_csv(path,index=False)
        self.assertEqual(self.run_eval()['successes'],1)

    def test_cli_prints_mixed_success_failure_and_exclusion_correctly(self):
        self.attempt();self.attempt((True,False));self.attempt();self.rows.pop()
        manifest=self.root/'manifest.csv'
        pd.DataFrame(self.rows).to_csv(manifest,index=False)
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            main(['--model',str(self.model),'--manifest',str(manifest),
                  '--output',str(self.root/'output'),'--data-kind','synthetic'])
        lines=output.getvalue().splitlines()
        self.assertIn('성공',lines[0]);self.assertIn('실패',lines[1]);self.assertIn('제외',lines[2])

    def test_reboot_rejected(self):
        self.attempt()
        def reset(f):
            f.loc[30:,'time_ms'] -= 100000
            return f
        self.change_raw(fn=reset)
        self.assertIn('nonmonotonic_clock_or_sequence',self.run_eval()['exclusion_reasons'])


if __name__=='__main__':
    unittest.main()
