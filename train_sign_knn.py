"""수어 웨어러블: 수집된 회차별 원시 센서 데이터로 KNN 분류 실험.

실행: python train_sign_knn.py
입력: data/sensor_dataset_v2.csv (원본은 읽기만 함)
출력: data/knn_results/ (특징, 평가, 모델)
주의: 한 사람에게만 수집된 수어는 사람/동작 혼동 가능성이 있습니다.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    print(f"필요한 라이브러리가 없습니다: {exc}\n설치: python -m pip install pandas numpy scikit-learn matplotlib joblib")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "sensor_dataset_v2.csv"
OUT = ROOT / "data" / "knn_results"
SENSORS = [*(f"fsr{i}" for i in range(1, 7)), "ax", "ay", "az", "gx", "gy", "gz"]
IDENTITY = ["subject_id", "session_id", "label", "hand", "trial_id"]
EXCLUDE_LABELS = {"시험", "연결시험", "test", "TEST"}
MIN_ROWS = 8  # 심하게 끊기거나 1~2샘플만 찍힌 회차는 제외


def build_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = set(IDENTITY + SENSORS + ["time_ms", "source"])
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"CSV에서 필요한 열이 없습니다: {', '.join(missing)}")

    raw = raw.copy()
    for col in IDENTITY + ["source"]:
        raw[col] = raw[col].astype("string").str.strip()
    raw = raw.loc[
        raw["source"].str.upper().eq("REAL").fillna(False)
        & raw["subject_id"].notna()
        & raw["subject_id"].ne("").fillna(False)
        & raw["label"].notna()
        & ~raw["label"].isin(EXCLUDE_LABELS)
        & raw["trial_id"].notna()
    ].copy()
    for col in SENSORS + ["time_ms"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["time_ms"])
    if raw.empty:
        raise ValueError("유효한 REAL 데이터가 없습니다. subject_labeler.py 실행 여부를 확인하세요.")

    # session_id explicitly separates repeated collection runs.
    records = []
    rejected = []
    for key, block in raw.groupby(IDENTITY, sort=False, dropna=False):
        record = dict(zip(IDENTITY, key))
        record["n_frames"] = len(block)
        record["start_ms"] = float(block["time_ms"].iloc[0])
        record["end_ms"] = float(block["time_ms"].iloc[-1])
        if len(block) < MIN_ROWS:
            rejected.append(record)
            continue
        for col in SENSORS:
            values = block[col].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            for statistic in ("mean", "std", "min", "max"):
                name = f"{col}_{statistic}"
                if finite.size == 0:
                    record[name] = np.nan
                elif statistic == "mean":
                    record[name] = float(np.mean(finite))
                elif statistic == "std":
                    record[name] = float(np.std(finite))
                elif statistic == "min":
                    record[name] = float(np.min(finite))
                else:
                    record[name] = float(np.max(finite))
        records.append(record)
    return pd.DataFrame(records), pd.DataFrame(rejected)


def main() -> int:
    if not INPUT.exists():
        print(f"입력 파일이 없습니다: {INPUT}")
        print("먼저 python prepare_sensor_v2.py 로 통합 CSV를 만드세요.")
        return 1

    features, rejected = build_features(pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False))
    if features.empty:
        print("학습 가능한 회차가 없습니다. 센서값을 확인하세요.")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUT / "trial_features.csv", index=False, encoding="utf-8-sig")
    if not rejected.empty:
        rejected.to_csv(OUT / "excluded_short_trials.csv", index=False, encoding="utf-8-sig")
    counts = features.groupby(["subject_id", "label"], dropna=False).size().unstack(fill_value=0)
    counts.to_csv(OUT / "subject_label_counts.csv", encoding="utf-8-sig")
    print("\n[참가자별 / 동작별 유효 회차 수]")
    print(counts.to_string())
    print(f"\n전체 학습 후보: {len(features)}회차 / 제외한 짧은 회차: {len(rejected)}개")

    labels = features["label"].astype(str)
    label_counts = labels.value_counts()
    if len(label_counts) < 2:
        print("\nKNN 분류에는 최소 2종류의 수어가 필요합니다. 특징 파일은 저장했습니다.")
        return 0
    if label_counts.min() < 4:
        print("\n동작별 회차가 4개 미만이라 학습/평가를 안전하게 나누기 어렵습니다.")
        return 0

    single_subject = counts.gt(0).sum(axis=0)
    if any(single_subject < 2):
        print("\n[주의] 일부 수어가 한 사람에게서만 측정됐습니다.")
        print("정확도가 높아도 실제 수어 동작보다 참가자 차이를 학습했을 수 있습니다.")
        print("두 사람이 동일한 수어들을 각각 측정하면 검증이 더 타당해집니다.")

    columns = [f"{c}_{stat}" for c in SENSORS for stat in ("mean", "std", "min", "max")]
    X = features[columns]
    idx_train, idx_test = train_test_split(
        np.arange(len(features)), test_size=0.25, random_state=42,
        stratify=labels,
    )
    k = min(3, len(idx_train))
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])
    model.fit(X.iloc[idx_train], labels.iloc[idx_train])
    predicted = model.predict(X.iloc[idx_test])
    accuracy = accuracy_score(labels.iloc[idx_test], predicted)
    classes = sorted(labels.unique().tolist())
    matrix = confusion_matrix(labels.iloc[idx_test], predicted, labels=classes)
    report = classification_report(labels.iloc[idx_test], predicted, labels=classes,
                                   zero_division=0)
    (OUT / "evaluation.txt").write_text(
        f"수어 동작 1회(5초) = 학습 샘플 1개\n"
        f"기준: 회차 단위 75% train / 25% test (random_state=42)\n"
        f"K={k}, 평가 정확도={accuracy:.4f} ({accuracy*100:.1f}%)\n"
        f"테스트 회차={len(idx_test)}\n\n{report}\n"
        "주의: 같은 참가자가 train/test 양쪽에 들어갈 수 있습니다. "
        "이 결과는 신규 참가자 일반화 성능이 아닙니다.\n",
        encoding="utf-8",
    )
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
        OUT / "confusion_matrix.csv", encoding="utf-8-sig"
    )
    try:
        fig, ax = plt.subplots(figsize=(5.5, 4.7))
        ax.imshow(matrix, cmap="Blues")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=11)
        ax.set_xticks(range(len(classes)), labels=[f"C{i+1}" for i in range(len(classes))])
        ax.set_yticks(range(len(classes)), labels=[f"C{i+1}" for i in range(len(classes))])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"KNN | k={k} | accuracy={accuracy:.1%}")
        fig.tight_layout()
        fig.savefig(OUT / "confusion_matrix.png", dpi=150)
        plt.close(fig)
        (OUT / "class_index.txt").write_text(
            "\n".join(f"C{i+1} = {name}" for i, name in enumerate(classes)), encoding="utf-8"
        )
    except Exception as exc:
        print(f"혼동행렬 그림 저장 실패(텍스트 CSV는 저장됨): {exc}")
    joblib.dump({"pipeline": model, "columns": columns, "classes": classes}, OUT / "knn_model.joblib")
    print(f"\nKNN 정확도: {accuracy:.1%} (시험 데이터 {len(idx_test)}회차, k={k})")
    print("※ 현재 정확도는 같은 참가자의 반복 데이터가 포함될 수 있는 초기 시험 결과입니다.")
    print(f"결과 저장: {OUT}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f"실행 오류: {exc}")
        sys.exit(1)
