"""Render a compact poster figure from saved real held-out predictions only."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    root = args.results.resolve()
    source = root / 'outer_predictions.csv'
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    result = json.loads((root / 'evaluation.json').read_text(encoding='utf-8'))['nested_selection']
    rows = pd.read_csv(source)
    rows = rows.loc[rows.method.eq('nested_selection')]
    labels = result['labels']
    assert not rows.file.duplicated().any()
    assert all(r.person not in r.train_people.split('|') for r in rows.itertuples())
    cm = pd.crosstab(rows.label, rows.prediction).reindex(index=labels, columns=labels, fill_value=0).to_numpy()
    assert np.array_equal(cm, result['confusion_matrix'])
    n, correct = int(cm.sum()), int(cm.trace())
    assert n == result['trials'] and correct == result['correct']
    rates = cm / cm.sum(axis=1, keepdims=True) * 100
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
                         'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(10, 9.3))
    plot = ax.imshow(rates, cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks(range(3), labels, fontsize=16)
    ax.set_yticks(range(3), labels, fontsize=16)
    ax.set_xlabel('예측한 수어', fontsize=17, labelpad=12)
    ax.set_ylabel('실제 수행한 수어', fontsize=17, labelpad=12)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{cm[i, j]}회\n({rates[i, j]:.1f}%)', ha='center', va='center',
                    fontsize=21, color='white' if rates[i, j] > 55 else '#123b60')
    bar = fig.colorbar(plot, ax=ax, fraction=.046, pad=.04)
    bar.set_label('실제 수어별 비율 (%)', fontsize=12)
    bar.ax.tick_params(labelsize=12)
    fig.suptitle('XIAO 삼키다·아니다·상처\n탐색적 내부 교차검증', fontsize=23,
                 fontweight='bold', y=.98, linespacing=1.3)
    ax.set_title(f'{rows.person.nunique()}명 · {n}회 중 {correct}회 정답 · {correct/n:.2%}\n'
                 '양손 FSR+IMU · 참가자 1명씩 제외', fontsize=15, pad=16)
    fig.subplots_adjust(left=.13, right=.92, top=.78, bottom=.10)
    out = root / '공유자료'
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ['png', 'svg']:
        path = out / f'04_혼동행렬_포스터용.{suffix}'
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(path)
    plt.close(fig)
    caption = ('XIAO 양손 FSR+IMU를 이용한 참가자 제외 탐색적 내부 교차검증. '
               f'{n}회 중 {correct}회 정답({correct/n:.2%}). '
               '기존 결과를 참고해 선정한 수어이며 수어별 측정 날짜가 다르다. '
               '새 외부 데이터 검증 결과가 아니다. '
               '3종을 모두 측정한 참가자 8명과 일부 수어만 측정한 참가자 2명을 포함했다.\n')
    (out / '04_혼동행렬_포스터용_설명.txt').write_text(caption, encoding='utf-8')
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    print(f'Validated saved predictions: {correct}/{n}')


if __name__ == '__main__':
    main()
