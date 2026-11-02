"""Pair two hands per trial without equating independent device clocks."""
from dataclasses import dataclass
import re

import numpy as np
import pandas as pd

from train_sign_knn import SENSORS

KEY = ['subject_id', 'session_id', 'trial_id']
STATS = ['mean', 'std', 'min', 'max']
BASE_FEATURES = [f'{sensor}_{stat}' for sensor in SENSORS for stat in STATS]
FEATURES = [f'{hand}_{name}' for hand in ('left', 'right') for name in BASE_FEATURES]


@dataclass(frozen=True)
class PairingPolicy:
    tolerance_ms: float = 150
    max_gap_ms: float = 500
    min_span_ms: float = 3000
    min_match_fraction: float = 0.7
    min_samples: int = 8
    allow_legacy_timing: bool = False

    def __post_init__(self):
        if (not all(np.isfinite(v) for v in (self.tolerance_ms, self.max_gap_ms,
                                            self.min_span_ms, self.min_match_fraction))
                or self.tolerance_ms < 0 or self.max_gap_ms <= 0
                or not 0 <= self.min_span_ms <= 5000
                or not 0 < self.min_match_fraction <= 1 or self.min_samples < 2):
            raise ValueError('Invalid pairing quality thresholds.')


def match_times(left, right, tolerance):
    """Nearest available right timestamp, monotonic and never reused."""
    pairs, j = [], 0
    for i, value in enumerate(left):
        while j < len(right) and right[j] < value - tolerance:
            j += 1
        candidates = [k for k in (j, j + 1) if k < len(right) and abs(right[k] - value) <= tolerance]
        if candidates:
            k = min(candidates, key=lambda k: abs(right[k] - value))
            pairs.append((i, k))
            j = k + 1
    return pairs


def summarize(block, prefix):
    result = {}
    for sensor in SENSORS:
        values = block[sensor].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        for stat in STATS:
            value = np.nan if not len(values) else float(getattr(np, stat)(values))
            result[f'{prefix}_{sensor}_{stat}'] = value
    return result


def build_paired_features(raw, policy=PairingPolicy()):
    required = set(KEY + ['label', 'hand', 'source', *SENSORS])
    if missing := required - set(raw.columns):
        raise ValueError(f'Missing columns: {sorted(missing)}')
    raw = raw.copy()
    for col in KEY + ['label', 'hand', 'source']:
        raw[col] = raw[col].astype('string').str.strip()
    raw = raw.loc[raw.source.eq('REAL') & ~raw.label.str.lower().isin(
        ['rest', 'test', '시험', '연결시험'])].copy()
    if raw[KEY + ['label', 'hand']].isna().any().any():
        raise ValueError('Missing participant/session/trial/label/hand.')
    if raw[KEY + ['label', 'hand']].eq('').any().any():
        raise ValueError('Empty identity metadata.')
    for col in SENSORS:
        raw[col] = pd.to_numeric(raw[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    invalid_sessions = set(raw.groupby('session_id').filter(
        lambda g: g.subject_id.nunique() != 1 or g.label.nunique() != 1
        or ('gesture_type' in g and g.gesture_type.dropna().nunique() > 1)).session_id)
    records, quality = [], []
    for key, block in raw.groupby(KEY, sort=True):
        q = dict(zip(KEY, key))
        q.update(status='rejected', reason='', left_samples=int(block.hand.eq('left').sum()),
                 right_samples=int(block.hand.eq('right').sum()), timing='unknown')
        try:
            if key[1] in invalid_sessions or block.label.nunique() != 1:
                raise ValueError('conflicting_session_metadata')
            label = str(block.label.iloc[0])
            trial_match = re.fullmatch(re.escape(label) + r'([1-9]|1[0-9]|20)', str(key[2]))
            if not trial_match:
                raise ValueError('invalid_trial_id')
            if set(block.hand) != {'left', 'right'}:
                raise ValueError('missing_or_invalid_hand')
            if 'sample_seq' in block:
                sequences = pd.to_numeric(block.sample_seq, errors='coerce')
                if sequences.isna().any() or (sequences < 1).any() or (sequences % 1 != 0).any():
                    raise ValueError('invalid_sample_sequence')
                block = block.assign(sample_seq=sequences)
                if block.duplicated(['hand', 'sample_seq']).any():
                    raise ValueError('duplicate_sample_key')
            mode = 'unknown'
            if 'gesture_type' in block:
                modes = block.gesture_type.dropna().unique()
                if len(modes) != 1 or block.gesture_type.isna().any():
                    raise ValueError('conflicting_gesture_type')
                mode = str(modes[0])
                if mode not in {'left_only', 'right_only', 'both'}:
                    raise ValueError('invalid_gesture_type')
            q['gesture_type'] = mode
            hands = [block.loc[block.hand.eq(h)].copy() for h in ('left', 'right')]
            if min(map(len, hands)) < policy.min_samples:
                raise ValueError('too_few_samples')
            timed = {'host_elapsed_ms', 'trial_elapsed_ms'}.issubset(block.columns)
            if not timed:
                if not policy.allow_legacy_timing:
                    raise ValueError('timing_unknown')
                q['timing'] = 'legacy_unverified'
                matched = hands
            else:
                if mode == 'unknown':
                    raise ValueError('missing_gesture_type')
                times = []
                for side, hand in enumerate(hands):
                    for col in ('trial_elapsed_ms', 'host_elapsed_ms'):
                        hand[col] = pd.to_numeric(hand[col], errors='coerce')
                    if not np.isfinite(hand[['trial_elapsed_ms', 'host_elapsed_ms']].to_numpy()).all():
                        raise ValueError('invalid_timing')
                    t = hand.trial_elapsed_ms.to_numpy()
                    expected = hand.host_elapsed_ms.to_numpy() - (int(trial_match[1]) - 1) * 8000
                    if (t < 0).any() or (t >= 5000).any() or not np.allclose(t, expected, atol=0.01, rtol=0):
                        raise ValueError('clock_trial_mismatch')
                    hand = hand.sort_values('trial_elapsed_ms')
                    hands[side] = hand
                    t = hand.trial_elapsed_ms.to_numpy()
                    if len(np.unique(t)) != len(t):
                        raise ValueError('duplicate_timestamps')
                    if t[-1] - t[0] < policy.min_span_ms or t[0] > 1000 or t[-1] < 4000:
                        raise ValueError('partial_trial_window')
                    if np.diff(t).max() > policy.max_gap_ms:
                        raise ValueError('large_gap')
                    if 'time_ms' in hand:
                        device_time = pd.to_numeric(hand.time_ms, errors='coerce').to_numpy()
                        if not np.isfinite(device_time).all() or (np.diff(device_time) < 0).any():
                            raise ValueError('device_clock_reset_or_invalid')
                        # Compare durations only; each board has its own boot epoch.
                        if abs(np.ptp(device_time) - np.ptp(t)) > policy.max_gap_ms:
                            raise ValueError('device_host_duration_mismatch')
                    times.append(t)
                pairs = match_times(*times, policy.tolerance_ms)
                fraction = len(pairs) / max(map(len, hands))
                q.update(matched_samples=len(pairs), matched_fraction=fraction,
                         left_span_ms=float(np.ptp(times[0])), right_span_ms=float(np.ptp(times[1])))
                if len(pairs) < policy.min_samples or fraction < policy.min_match_fraction:
                    raise ValueError('insufficient_temporal_overlap')
                matched = [hands[side].iloc[[p[side] for p in pairs]] for side in (0, 1)]
                if min(np.ptp(h.trial_elapsed_ms.to_numpy()) for h in matched) < policy.min_span_ms:
                    raise ValueError('partial_matched_window')
                q.update(timing='host_receipt_verified',
                         max_pair_delta_ms=float(max(abs(times[0][i] - times[1][j]) for i, j in pairs)))
            if any(hand[SENSORS].isna().all().any() for hand in matched):
                raise ValueError('entire_sensor_missing')
            if any(not np.isfinite(hand[[f'fsr{i}' for i in range(1, 7)]].to_numpy()).all() for hand in matched):
                raise ValueError('invalid_fsr')
            if any(((hand[[f'fsr{i}' for i in range(1, 7)]] < 0)
                    | (hand[[f'fsr{i}' for i in range(1, 7)]] > 4095)).any().any() for hand in matched):
                raise ValueError('invalid_fsr_range')
            record = dict(zip(KEY, key))
            record.update(label=label, gesture_type=mode)
            for hand, samples in zip(('left', 'right'), matched):
                record.update(summarize(samples, hand))
            records.append(record)
            q.update(status='accepted', reason='')
        except ValueError as error:
            q['reason'] = str(error)
        quality.append(q)
    return pd.DataFrame(records), pd.DataFrame(quality)


def mode_features(features, mode):
    if mode == 'both':
        return features[FEATURES].copy()
    if mode in ('left', 'right'):
        return features[[f'{mode}_{c}' for c in BASE_FEATURES]].set_axis(BASE_FEATURES, axis=1)
    if mode in ('active', 'passive'):
        if not features.gesture_type.isin(['left_only', 'right_only']).all():
            raise ValueError('Active/passive comparison requires one-handed signs.')
        use_left = features.gesture_type.eq('left_only').to_numpy()
        if mode == 'passive':
            use_left = ~use_left
        values = np.where(use_left[:, None], mode_features(features, 'left'), mode_features(features, 'right'))
        return pd.DataFrame(values, columns=BASE_FEATURES, index=features.index)
    raise ValueError(f'Unknown feature mode: {mode}')
