"""Label-independent, whole-trial sensor representations for OFFLINE comparison.

These are NOT rest/force/orientation calibration. Each completed ~5 s capture is
available at prediction time; no other trial or participant is used here.
"""
import numpy as np

STATS = ('mean', 'std', 'min', 'max')
FSR_SCALE_FLOOR = 50.0  # ADC counts; exploratory guard, not a health threshold


def validate(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 12 or len(values) < 8:
        raise ValueError('Expected at least eight rows of six FSR and six IMU channels')
    if not np.isfinite(values).all() or (values[:, :6] < 0).any():
        raise ValueError('Nonfinite sensor or negative FSR')
    return values


def summarize(values, robust=False):
    if robust:
        # Exactly four numbers per channel; separate feature schema from baseline.
        stats = [np.median(values, axis=0),
                 np.quantile(values, .75, axis=0)-np.quantile(values, .25, axis=0),
                 np.quantile(values, .05, axis=0), np.quantile(values, .95, axis=0)]
    else:
        stats = [values.mean(axis=0), values.std(axis=0),
                 values.min(axis=0), values.max(axis=0)]
    return np.stack(stats, axis=1).ravel()


def hand_features(values, *, scale_fsr=False, magnitude_imu=False, robust=False,
                  imu_only=False):
    values = validate(values)
    fsr = values[:, :6].copy()
    if scale_fsr:
        # One scale shared by six channels preserves pressure distribution and
        # time variation. Tiny/zero readings are never amplified above the floor.
        scale = max(float(np.quantile(fsr, .95)), FSR_SCALE_FLOOR)
        fsr /= scale
    imu = values[:, 6:]
    if magnitude_imu:
        imu = np.column_stack([np.linalg.norm(imu[:, :3], axis=1),
                               np.linalg.norm(imu[:, 3:], axis=1)])
    channels = imu if imu_only else np.column_stack([fsr, imu])
    return summarize(channels, robust=robust)


REPRESENTATIONS = {
    'baseline_all': {},
    'fsr_scale': {'scale_fsr': True},
    'imu_magnitude_all': {'magnitude_imu': True},
    'fsr_scale_imu_magnitude': {'scale_fsr': True, 'magnitude_imu': True},
    'robust_all': {'robust': True},
    'baseline_imu': {'imu_only': True},
    'imu_magnitude_only': {'imu_only': True, 'magnitude_imu': True},
}


def paired_features(hands, representation):
    if set(hands) != {'left', 'right'}:
        raise ValueError('Both hands required; never fill missing sensors with zero')
    params = REPRESENTATIONS[representation]
    return np.concatenate([hand_features(hands[h], **params) for h in ('left', 'right')])
