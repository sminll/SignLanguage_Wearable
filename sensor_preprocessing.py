"""Completed-trial preprocessing for XIAO band recordings.

No raw files are modified. Rest subtraction is a session baseline, NOT physical
force calibration or accelerometer orientation/gravity calibration. Both device
clocks stay independent. All learned feature selection belongs inside fit().
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

SENSORS = [f'fsr{i}' for i in range(1, 7)] + ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
STATS = ['mean', 'std', 'min', 'max']
FEATURES = [f'{h}_{s}_{stat}' for h in ['left', 'right'] for s in SENSORS for stat in STATS]
REST_LABELS = {'휴식', 'rest', 'resting', 'neutral'}


@dataclass(frozen=True)
class Policy:
    min_samples: int = 8
    min_span_ms: float = 3000
    max_gap_ms: float = 500
    rest_min_samples: int = 30
    rest_min_span_ms: float = 4000
    rest_max_age_seconds: float = 1800
    rest_max_gyro_std_dps: float = 3
    rest_max_gyro_bias_dps: float = 20
    rest_max_accel_std_g: float = .03
    filter_strength: float = .5
    filter_max_span_ms: float = 350
    fsr_scale_floor_adc: float = 50

    def __post_init__(self):
        values = list(asdict(self).values())
        if (not np.isfinite(values).all() or self.min_samples < 3 or self.rest_min_samples < 3
                or int(self.min_samples) != self.min_samples or int(self.rest_min_samples) != self.rest_min_samples
                or self.min_span_ms <= 0 or self.max_gap_ms <= 0 or self.rest_min_span_ms <= 0
                or self.rest_max_age_seconds <= 0 or self.filter_max_span_ms <= 0
                or not 0 <= self.filter_strength <= 1 or self.fsr_scale_floor_adc <= 0
                or min(self.rest_max_gyro_std_dps, self.rest_max_gyro_bias_dps, self.rest_max_accel_std_g) <= 0):
            raise ValueError('Invalid preprocessing policy')


@dataclass(frozen=True)
class Recipe:
    rest: bool = False
    denoise: bool = False
    normalize: bool = False
    select_features: bool = False


@dataclass
class HandCapture:
    account: str
    person: str
    session_id: str
    trial_id: str
    label: str
    kind: str
    hand: str
    board: str
    calibration_version: str
    source_path: str
    source_sha256: str
    start_utc: str
    end_utc: str
    calibration_verified: bool
    times_ms: np.ndarray
    values: np.ndarray

    @property
    def scope(self):
        return (self.account, self.person, self.session_id, self.hand,
                self.board, self.calibration_version)


@dataclass(frozen=True)
class RestProfile:
    scope: tuple
    source_path: str
    source_sha256: str
    start_utc: str
    end_utc: str
    device_end_ms: float
    fsr_median: tuple
    gyro_median: tuple
    fsr_noise_mad: tuple
    source_calibration_verified: bool
    samples: int
    span_ms: float
    gyro_std_dps: float
    gyro_bias_dps: float
    accel_std_g: float

    def __post_init__(self):
        if (len(self.scope) != 6 or any(not isinstance(v,str) or not v for v in self.scope)
                or self.scope[3] not in {'left','right'} or self.scope[4] != 'XIAO'
                or len(self.fsr_median) != 6 or len(self.gyro_median) != 3 or len(self.fsr_noise_mad) != 6
                or not re.fullmatch(r'[0-9a-f]{64}', self.source_sha256)
                or not np.isfinite([self.device_end_ms,*self.fsr_median,*self.gyro_median,*self.fsr_noise_mad,
                                    self.samples,self.span_ms,self.gyro_std_dps,self.gyro_bias_dps,self.accel_std_g]).all()
                or self.device_end_ms < 0 or min(self.fsr_median) < 0 or max(self.fsr_median) > 4095
                or min(self.fsr_noise_mad) < 0 or not isinstance(self.source_calibration_verified,bool)
                or self.samples < 3 or int(self.samples) != self.samples or self.span_ms <= 0
                or min(self.gyro_std_dps,self.gyro_bias_dps,self.accel_std_g) < 0):
            raise ValueError('Invalid rest profile schema')
        start, end = pd.Timestamp(self.start_utc), pd.Timestamp(self.end_utc)
        if start.tzinfo is None or end.tzinfo is None or pd.isna(start) or pd.isna(end) or start > end:
            raise ValueError('Invalid UTC rest interval')

    @property
    def profile_id(self):
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False).encode('utf-8')
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self):
        return {**asdict(self), 'profile_id': self.profile_id,
                'quality_status': 'statistical_rest_checks_passed',
                'physical_force_calibrated': False}

    @classmethod
    def from_dict(cls, record):
        fields = {k:record[k] for k in cls.__dataclass_fields__}
        for key in ['scope','fsr_median','gyro_median','fsr_noise_mad']:
            fields[key] = tuple(fields[key])
        profile = cls(**fields)
        if record.get('profile_id') != profile.profile_id:
            raise ValueError('Rest profile contents/hash mismatch')
        return profile


def normalize_person(value):
    match = re.fullmatch(r'subject-?(\d+)', str(value).strip(), re.I)
    return 'subject'+str(int(match[1])) if match else str(value).strip()


def read_capture(frame, *, account, source_path='', source_sha256='', policy=Policy()):
    """Read current app CSV schema. Raw source must be REAL XIAO sensor values."""
    required = {'subject_id','session_id','trial_id','label','record_kind','board_type',
                'calibration_version','source','hand','time','time_ms','sample_seq','imu_ok', *SENSORS}
    if missing := required-set(frame.columns):
        raise ValueError('Missing raw columns: '+', '.join(sorted(missing)))
    if not str(account).strip() or frame.empty:
        raise ValueError('Explicit account and nonempty raw capture required')
    identity = {}
    for col in ['subject_id','session_id','trial_id','label','record_kind','board_type','calibration_version']:
        values = frame[col].astype(str).str.strip()
        if values.nunique() != 1 or not values.iloc[0] or values.iloc[0].lower() in {'nan','none'}:
            raise ValueError('Conflicting/empty '+col)
        identity[col] = values.iloc[0]
    if (not frame.source.astype(str).str.upper().eq('REAL').all() or identity['board_type'] != 'XIAO'
            or identity['record_kind'] not in {'training','calibration'}):
        raise ValueError('Only raw REAL XIAO training/calibration captures supported')
    if not set(frame.hand) <= {'left', 'right'} or not set(frame.hand):
        raise ValueError('Invalid hand')
    result = {}
    for hand, part in frame.groupby('hand', sort=False):
        values = part[SENSORS].to_numpy(dtype=float, copy=True)
        times = pd.to_numeric(part.time_ms, errors='raise').to_numpy(float)
        sequence = pd.to_numeric(part.sample_seq, errors='raise').to_numpy(float)
        if (len(part) < policy.min_samples or not np.isfinite(values).all()
                or not np.isfinite(times).all() or not np.isfinite(sequence).all()
                or (values[:, :6] < 0).any() or (values[:, :6] > 4095).any()):
            raise ValueError('Insufficient/nonfinite/out-of-range raw sensors')
        if (np.diff(times) <= 0).any() or (np.diff(sequence) <= 0).any():
            raise ValueError('Device clock/sample sequence is not increasing')
        if times[-1]-times[0] < policy.min_span_ms or np.diff(times).max() > policy.max_gap_ms:
            raise ValueError('Short/gapped capture; never silently interpolate')
        if not part.imu_ok.astype(str).str.lower().isin(['1','true','1.0']).all():
            raise ValueError('IMU validity flag is false')
        host = pd.to_datetime(part.time, format='mixed', errors='raise', utc=True)
        if host.isna().any() or not host.is_monotonic_increasing:
            raise ValueError('Invalid UTC receipt timestamps')
        verified = ('calibration_verified' in part and
                    part.calibration_verified.astype(str).str.lower().isin(['true','1','1.0']).all())
        values.setflags(write=False)
        times.setflags(write=False)
        result[hand] = HandCapture(str(account), normalize_person(identity['subject_id']),
            identity['session_id'], identity['trial_id'], identity['label'], identity['record_kind'],
            hand, identity['board_type'], identity['calibration_version'], str(source_path),
            source_sha256, host.min().isoformat(), host.max().isoformat(), bool(verified), times, values)
    return result


def make_rest_profile(capture, policy=Policy()):
    if capture.kind != 'calibration' or capture.label.strip().lower() not in REST_LABELS:
        raise ValueError('An explicit rest calibration capture is required')
    if len(capture.values) < policy.rest_min_samples or np.ptp(capture.times_ms) < policy.rest_min_span_ms:
        raise ValueError('Rest should contain approximately five seconds of samples')
    gyro = capture.values[:, 9:12]
    accel = capture.values[:, 6:9]
    gyro_std = float(np.linalg.norm(gyro.std(axis=0)))
    gyro_bias = float(np.linalg.norm(np.median(gyro,axis=0)))
    accel_std = float(np.linalg.norm(accel.std(axis=0)))
    if (gyro_std > policy.rest_max_gyro_std_dps or gyro_bias > policy.rest_max_gyro_bias_dps
            or accel_std > policy.rest_max_accel_std_g):
        raise ValueError('Rest motion too large for the configured baseline policy')
    fsr = capture.values[:, :6]
    median = np.median(fsr, axis=0)
    return RestProfile(capture.scope, capture.source_path, capture.source_sha256,
        capture.start_utc, capture.end_utc, float(capture.times_ms[-1]), tuple(median),
        tuple(np.median(gyro, axis=0)), tuple(np.median(np.abs(fsr-median), axis=0)),
        capture.calibration_verified,len(capture.values),float(np.ptp(capture.times_ms)),
        gyro_std,gyro_bias,accel_std)


def match_rest(capture, profiles, policy=Policy()):
    candidates = []
    for profile in profiles:
        if profile.scope != capture.scope:
            continue
        if (profile.samples < policy.rest_min_samples or profile.span_ms < policy.rest_min_span_ms
                or profile.gyro_std_dps > policy.rest_max_gyro_std_dps
                or profile.gyro_bias_dps > policy.rest_max_gyro_bias_dps
                or profile.accel_std_g > policy.rest_max_accel_std_g):
            continue
        age = (pd.Timestamp(capture.start_utc)-pd.Timestamp(profile.end_utc)).total_seconds()
        # Comparing device boot epochs is allowed ONLY within the same hand scope.
        if 0 <= age <= policy.rest_max_age_seconds and capture.times_ms[0] >= profile.device_end_ms:
            candidates.append(profile)
    if not candidates:
        raise ValueError('No matching previous rest for this account/person/session/hand/version')
    newest = max(pd.Timestamp(p.end_utc) for p in candidates)
    best = {p.profile_id:p for p in candidates if pd.Timestamp(p.end_utc) == newest}
    if len(best) != 1:
        raise ValueError('Ambiguous previous rest profiles')
    return next(iter(best.values()))


def weak_median_filter(values, times_ms, policy=Policy()):
    """Half blend a three-sample median; never cross a trial/hand or large gap.

    Endpoints stay unchanged. At nominal 10 Hz the interior window spans ~0.2s.
    This is completed-window processing, not a validated streaming filter.
    """
    values = np.asarray(values, float)
    times_ms = np.asarray(times_ms, float)
    if (values.ndim != 2 or len(values) != len(times_ms) or len(values) < 3
            or not np.isfinite(values).all() or not np.isfinite(times_ms).all()
            or (np.diff(times_ms) <= 0).any()):
        raise ValueError('Invalid finite time-ordered filter input')
    result = values.copy()
    for i in range(1, len(values)-1):
        if times_ms[i+1]-times_ms[i-1] <= policy.filter_max_span_ms:
            median = np.median(values[i-1:i+2], axis=0)
            result[i] += policy.filter_strength*(median-values[i])
    return result


def process_hand(capture, recipe, profiles=(), policy=Policy()):
    if capture.kind != 'training':
        raise ValueError('Only training captures can be transformed for prediction')
    values = capture.values.copy()
    profile = None
    if recipe.rest:
        profile = match_rest(capture, profiles, policy)
        values[:, :6] -= np.array(profile.fsr_median)
        values[:, 9:12] -= np.array(profile.gyro_median)
        # Acceleration keeps gravity/pose; negative pressure DELTAS are valid.
    if recipe.denoise:
        values = weak_median_filter(values, capture.times_ms, policy)
    scale = 1.0
    if recipe.normalize:
        scale = max(float(np.quantile(np.abs(values[:, :6]), .95)), policy.fsr_scale_floor_adc)
        values[:, :6] /= scale
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite processed sensors')
    return values, dict(profile_id=profile.profile_id if profile else '', fsr_scale=scale,
        profile_source_verified=profile.source_calibration_verified if profile else None,
        rest_applied=bool(profile), denoise_applied=recipe.denoise, normalized=recipe.normalize)


def extract_features(hands, recipe=Recipe(), profiles=(), policy=Policy()):
    if set(hands) != {'left','right'}:
        raise ValueError('Both hands required; missing hand is never replaced with zeros')
    identities = {(c.account,c.person,c.session_id,c.trial_id,c.label,c.kind) for c in hands.values()}
    if len(identities) != 1 or next(iter(hands.values())).kind != 'training':
        raise ValueError('Conflicting identity or non-training trial')
    result, provenance = {}, []
    for hand in ['left','right']:
        if hands[hand].hand != hand:
            raise ValueError('Hand mapping conflicts with capture')
        values, log = process_hand(hands[hand], recipe, profiles, policy)
        for i, sensor in enumerate(SENSORS):
            for stat in STATS:
                result[f'{hand}_{sensor}_{stat}'] = float(getattr(np, stat)(values[:, i]))
        provenance.append(dict(hand=hand, **log))
    return result, provenance


class TrainingFeatureSelector(TransformerMixin, BaseEstimator):
    """Constant/correlation pruning then stable ANOVA ranking, fitted on train only."""
    def __init__(self, max_features=32, correlation_threshold=.98):
        self.max_features = max_features
        self.correlation_threshold = correlation_threshold

    def fit(self, X, y):
        values = np.asarray(X, dtype=float)
        labels = np.asarray(y)
        if (values.ndim != 2 or len(values) != len(labels) or not np.isfinite(values).all()
                or len(np.unique(labels)) < 2 or self.max_features < 1
                or int(self.max_features) != self.max_features or not 0 < self.correlation_threshold <= 1):
            raise ValueError('Invalid feature selection input/settings')
        self.n_features_in_ = values.shape[1]
        self.training_rows_ = len(values)
        self.training_mean_ = values.mean(axis=0)
        std = values.std(axis=0)
        active = np.flatnonzero(std > 1e-12)
        if len(active) == 0:
            raise ValueError('No varying training features')
        z = (values[:, active]-values[:, active].mean(axis=0))/std[active]
        correlations = np.abs(z.T@z/len(z))
        kept_local = []
        for i in range(len(active)):
            if not any(correlations[i, j] >= self.correlation_threshold for j in kept_local):
                kept_local.append(i)
        candidates = active[kept_local]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            # Synthetic perfect class separation can have zero within-class
            # variance. Infinite scores are handled deterministically below.
            warnings.filterwarnings('ignore', message=r'Features .* are constant\.', category=UserWarning)
            scores, _ = f_classif(values[:, candidates], labels)
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(float).max, neginf=-np.inf)
        ranked = sorted(range(len(candidates)), key=lambda i:(-scores[i], int(candidates[i])))
        self.selected_indices_ = np.sort(candidates[ranked[:self.max_features]])
        self.scores_ = np.full(values.shape[1], np.nan)
        self.scores_[candidates] = scores
        self.correlation_survivors_ = candidates
        return self

    def transform(self, X):
        check_is_fitted(self, 'selected_indices_')
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_ or not np.isfinite(values).all():
            raise ValueError('Prediction feature schema/nonfinite mismatch')
        return values[:, self.selected_indices_]

    def get_support(self):
        check_is_fitted(self, 'selected_indices_')
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_indices_] = True
        return mask


def make_pipeline(config, recipe):
    steps = [('scaler', StandardScaler())]
    if recipe.select_features:
        steps.append(('selector', TrainingFeatureSelector()))
    steps.append(('knn', KNeighborsClassifier(n_neighbors=int(config['k']), p=int(config['p']),
                  weights=config['weights'], algorithm='brute', n_jobs=1)))
    return Pipeline(steps)


def fit_bundle(features, labels, *, recipe, policy=Policy(), config=None, metadata=None):
    """Fit a separate candidate. This never installs/replaces an existing model."""
    config = dict(k=3,p=1,weights='uniform') if config is None else config
    frame = pd.DataFrame(features).loc[:,FEATURES]
    pipeline = make_pipeline(config,recipe).fit(frame,labels)
    return dict(schema='xiao_preprocessing_v1', recipe=asdict(recipe), policy=asdict(policy),
        feature_columns=FEATURES.copy(), pipeline=pipeline,
        classes=list(pipeline.named_steps['knn'].classes_),
        metadata={**(metadata or {}), 'external_accuracy':None,
                  'external_accuracy_display':'미측정', 'deployed':False})


def predict_capture(bundle, hands, profiles=()):
    """Apply the EXACT stored preprocessing before predicting a completed trial."""
    if bundle.get('schema') != 'xiao_preprocessing_v1' or bundle.get('feature_columns') != FEATURES:
        raise ValueError('Unsupported preprocessing model schema')
    recipe,policy = Recipe(**bundle['recipe']),Policy(**bundle['policy'])
    values,provenance = extract_features(hands,recipe,profiles,policy)
    frame = pd.DataFrame([values],columns=FEATURES)
    prediction = bundle['pipeline'].predict(frame)[0]
    return str(prediction),provenance
