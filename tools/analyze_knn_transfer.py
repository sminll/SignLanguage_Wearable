"""Post-hoc diagnostics, NOT an independent model selection score."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix

from train_sign_knn import ROOT, SENSORS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'data/knn_results/2026-11-02_subject1_subject4_to_subject2_retake')
    args = parser.parse_args()
    old_dir = ROOT / 'data/knn_results/2026-11-02_subject1_to_subject4/combined_model'
    train = pd.read_csv(old_dir / 'trial_features.csv', encoding='utf-8-sig')
    test = pd.read_csv(args.run / 'external_evaluation/trial_features.csv', encoding='utf-8-sig')
    raw = pd.read_csv(args.run / 'external_evaluation/action_samples.csv', encoding='utf-8-sig')
    baseline = joblib.load(old_dir / 'knn_model.joblib')
    pipeline = baseline['pipeline']
    columns = baseline['columns']
    classes = baseline['classes']
    assert len(train) == 120 and len(test) == 60
    assert set(train.subject_id) == {'subject1', 'subject4'}
    assert set(test.subject_id) == {'subject2'}
    assert not set(train.session_id) & set(test.session_id)
    assert set(raw.label) == set(classes)
    out = args.run / 'diagnostics'
    out.mkdir(exist_ok=False)
    feature_groups = {
        'FSR+IMU': columns,
        'FSR_only': [c for c in columns if c.startswith('fsr')],
        'IMU_only': [c for c in columns if not c.startswith('fsr')],
        'acceleration_only': [c for c in columns if c.startswith(('ax_', 'ay_', 'az_'))],
        'gyro_only': [c for c in columns if c.startswith(('gx_', 'gy_', 'gz_'))],
        'FSR1_2_4+IMU': [c for c in columns if not c.startswith(('fsr3_', 'fsr5_', 'fsr6_'))],
    }
    results = []
    for name, subset in feature_groups.items():
        model = clone(pipeline)
        model.fit(train[subset], train.label)
        pred = model.predict(test[subset])
        score = accuracy_score(test.label, pred)
        results.append({'features': name, 'n_features': len(subset),
                        'correct': int((test.label == pred).sum()), 'trials': len(test),
                        'accuracy': score,
                        'confusion_matrix': confusion_matrix(test.label, pred, labels=classes).tolist()})
    pd.DataFrame([{k: v for k, v in r.items() if k != 'confusion_matrix'} for r in results]).to_csv(
        out / 'posthoc_feature_comparisons.csv', index=False, encoding='utf-8-sig')
    imputer = pipeline.named_steps['imputer']
    scaler = pipeline.named_steps['scaler']
    scaled_train = scaler.transform(imputer.transform(train[columns]))
    scaled_test = scaler.transform(imputer.transform(test[columns]))
    assert np.allclose(scaled_train, pipeline.named_steps['knn']._fit_X)
    distances, neighbors = pipeline.named_steps['knn'].kneighbors(scaled_test)
    delta = (scaled_test - scaled_train[neighbors[:, 0]]) ** 2
    contributions = pd.DataFrame({'feature': columns,
                                  'mean_nearest_squared_distance': delta.mean(axis=0),
                                  'train_scale': scaler.scale_,
                                  'train_mean': scaler.mean_,
                                  'test_mean': test[columns].mean().to_numpy()})
    contributions['distance_share'] = contributions.mean_nearest_squared_distance / contributions.mean_nearest_squared_distance.sum()
    contributions.sort_values('distance_share', ascending=False).to_csv(
        out / 'nearest_neighbor_feature_contributions.csv', index=False, encoding='utf-8-sig')
    pd.concat([train, test]).groupby('subject_id')[[c + '_mean' for c in SENSORS]].median().to_csv(
        out / 'participant_feature_medians.csv', encoding='utf-8-sig')
    quality = {}
    for c in SENSORS:
        x = pd.to_numeric(raw[c], errors='coerce')
        quality[c] = {'missing': int(x.isna().sum()), 'min': float(x.min()), 'max': float(x.max()),
                      'zero_fraction': float(x.eq(0).mean())}
    accel_norm = np.sqrt(raw.ax ** 2 + raw.ay ** 2 + raw.az ** 2)
    gyro_norm = np.sqrt(raw.gx ** 2 + raw.gy ** 2 + raw.gz ** 2)
    quality['acceleration_norm'] = {'median': float(accel_norm.median()), 'min': float(accel_norm.min()), 'max': float(accel_norm.max())}
    quality['gyro_norm'] = {'median': float(gyro_norm.median()), 'max': float(gyro_norm.max())}
    diagnostic = {'note': 'Post-hoc on already evaluated subject2. Do not report as an independent test score or tune deployment on this alone.',
                  'classes': classes, 'feature_comparisons': results, 'test_sensor_quality': quality,
                  'top_distance_features': contributions.sort_values('distance_share', ascending=False).head(10).to_dict('records')}
    (out / 'analysis.json').write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(diagnostic, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
