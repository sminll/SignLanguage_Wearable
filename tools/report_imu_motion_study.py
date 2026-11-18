"""Report real fixed-cohort magnitude/alignment ablations and their limits."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import group_model_study as base


NAMES={
 'fixed_existing_knn':'기존 고정 IMU KNN',
 'previous_svm_replay':'이전 SVM+시간 특징 재현',
 'svm_original':'기존 시간 특징 · 내부 재선택',
 'svm_magnitude':'IMU 크기 추가 · 내부 선택',
 'svm_aligned':'동작 구간 정렬 · 내부 선택',
 'svm_magnitude_aligned':'크기+정렬 · 내부 선택',
 'train_selected_features':'특징 종류까지 학습 쪽에서 선택',
 'fixed_settings_magnitude':'이전 설정 고정 · 크기 추가',
 'fixed_settings_aligned':'이전 설정 고정 · 구간 정렬',
 'fixed_settings_magnitude_aligned':'이전 설정 고정 · 크기+정렬',
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Fresh output directory required')
    results,out=args.results,args.out
    summary=pd.read_csv(results/'summary.csv').set_index('method')
    predictions=pd.read_csv(results/'predictions.csv')
    metrics=json.loads((results/'metrics.json').read_text(encoding='utf-8'))
    for name,row in summary.iterrows():
        data=predictions[predictions.method.eq(name)]
        assert len(data)==row.trials==1170 and data.file.nunique()==1170
        assert int(data.label.eq(data.prediction).sum())==row.correct
        assert not any(r.person in r.training_people.split('|') for r in data.itertuples())
    out.mkdir(parents=True)
    for name in ['summary.csv','predictions.csv','per_sign.csv','per_person.csv','per_fold.csv','metrics.json',
                 'cohort_manifest.csv','selected_configs.csv','split_manifest.json','plan.json','verification.json',
                 'alignment_audit.csv',*[f'outer_{i}_selected_before_test.json' for i in range(1,6)]]:
        shutil.copyfile(results/name,out/name)
    font=Path('C:/Windows/Fonts/malgun.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    order=['fixed_existing_knn','previous_svm_replay','svm_magnitude','svm_aligned',
           'svm_magnitude_aligned','train_selected_features']
    fig,ax=plt.subplots(figsize=(11,5))
    bars=ax.barh([NAMES[k] for k in order],summary.loc[order,'accuracy']*100,
                 color=['#929daa','#667e94','#497eab','#497eab','#497eab','#28725b'])
    for bar,name in zip(bars,order):
        r=summary.loc[name]
        ax.text(bar.get_width()+.8,bar.get_y()+bar.get_height()/2,
                f'{int(r.correct)}/1170 · {r.accuracy:.2%}',va='center')
    ax.invert_yaxis();ax.set_xlim(0,100)
    ax.set_xlabel('전체 회차 정답률 (%)')
    ax.set_title('XIAO 6개 수어 · 10명 · 같은 사람별 5-fold\n설정 선택은 학습 쪽의 사람별 3-fold에서만 수행')
    fig.tight_layout();fig.savefig(out/'accuracy_comparison.png',dpi=240);fig.savefig(out/'accuracy_comparison.pdf');plt.close(fig)
    primary=next(r for r in metrics['results'] if r['method']=='train_selected_features')
    cm=np.asarray(primary['confusion_matrix'])
    ratios=cm/cm.sum(axis=1,keepdims=True)*100
    fig,ax=plt.subplots(figsize=(8,7))
    im=ax.imshow(ratios,cmap='Blues',vmin=0,vmax=100)
    ax.set_xticks(range(6),metrics['labels'],rotation=30,ha='right');ax.set_yticks(range(6),metrics['labels'])
    for (i,j),n in np.ndenumerate(cm):
        ax.text(j,i,f'{n}회\n({ratios[i,j]:.1f}%)',ha='center',va='center',color='white' if ratios[i,j]>55 else '#173a5e')
    ax.set_xlabel('예측 수어');ax.set_ylabel('실제 수어')
    ax.set_title(f"특징 종류까지 학습 쪽에서 선택한 절차\n{primary['correct']}/1170 · {primary['accuracy']:.2%}")
    fig.colorbar(im,ax=ax,label='실제 수어별 비율 (%)');fig.tight_layout()
    fig.savefig(out/'selected_procedure_confusion.png',dpi=240);plt.close(fig)
    fixed_order=['previous_svm_replay','fixed_settings_magnitude','fixed_settings_aligned',
                 'fixed_settings_magnitude_aligned','train_selected_features']
    fig,ax=plt.subplots(figsize=(11,4.5))
    bars=ax.barh([NAMES[k] for k in fixed_order],summary.loc[fixed_order,'accuracy']*100,
        color=['#667e94','#497eab','#497eab','#497eab','#28725b'])
    for bar,name in zip(bars,fixed_order):
        r=summary.loc[name]
        ax.text(bar.get_width()+.8,bar.get_y()+bar.get_height()/2,
                f'{int(r.correct)}/1170 · {r.accuracy:.2%}',va='center')
    ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('전체 회차 정답률 (%)')
    ax.set_title('이전 SVM 설정을 고정한 특징 변경 비교\n같은 6개 수어 · 10명 · 사람별 5-fold / 초록 행은 별도 내부 선택 절차')
    fig.tight_layout();fig.savefig(out/'fixed_settings_comparison.png',dpi=240);plt.close(fig)
    audit=pd.read_csv(results/'alignment_audit.csv')
    status=audit.groupby(['recipe','hand','status']).size().rename('hand_windows').reset_index()
    status.to_csv(out/'alignment_status.csv',index=False,encoding='utf-8-sig')
    rows=['| 방법 | 정답 / 시험 회차 | 정확도 | 이전 SVM 대비 |','|---|---:|---:|---:|']
    old=summary.loc['previous_svm_replay']
    for name in NAMES:
        r=summary.loc[name]
        rows.append(f'| {NAMES[name]} | {int(r.correct)}/1170 | {r.accuracy:.2%} | {(r.accuracy-old.accuracy)*100:+.2f}%p |')
    by_sign=pd.read_csv(results/'per_sign.csv')
    display=by_sign[by_sign.method.isin(['previous_svm_replay','svm_magnitude','svm_aligned',
        'svm_magnitude_aligned','train_selected_features'])].pivot(index='label',columns='method',values='mean')
    display.to_csv(out/'per_sign_comparison.csv',encoding='utf-8-sig')
    sign_rows=['| 수어 | 이전 SVM | 크기 추가 | 구간 정렬 | 둘 다 | 내부에서 특징 선택 |','|---|---:|---:|---:|---:|---:|']
    for label,r in display.iterrows():
        sign_rows.append(f'| {label} | {r.previous_svm_replay:.2%} | {r.svm_magnitude:.2%} | {r.svm_aligned:.2%} | {r.svm_magnitude_aligned:.2%} | {r.train_selected_features:.2%} |')
    raw=audit[audit.recipe.eq('raw')]
    good=int(raw.resampled.sum())
    chosen=summary.loc['train_selected_features']
    text=f'''# 2026-11-18 IMU 크기·동작 구간 정렬 비교

기존 데이터·모델을 유지한 별도 `feature/imu-motion-features` 브랜치의 실험이다. 이전과 같은 6개 수어(괜찮다·아니다·무엇·마시다·화장실·싫다), 10명, 1,170회를 사용했다. 오늘 들어온 다른 수어를 섞지 않았다.

## 결과

'''+ '\n'.join(rows)+f'''

사전 지정한 주된 비교 절차는 ‘특징 종류까지 학습 쪽에서 선택’이다. 이전 SVM 대비 정답 {int(chosen.correct-old.correct):+d}회, {(chosen.accuracy-old.accuracy)*100:+.2f}%p였다. 각 특징을 고정한 별도 비교 행 중 점수가 높은 것을 사후에 골라 새 데이터의 확정 성능이라고 부르지 않는다. 바깥 시험 사람의 정답은 내부 설정 선택에서 제외했다.

이전 설정을 유지한 크기+정렬 통제 비교는 {int(summary.loc['fixed_settings_magnitude_aligned','correct'])}/1170={summary.loc['fixed_settings_magnitude_aligned','accuracy']:.2%}였다. 이 값은 특징별 설정을 다시 고른 ‘크기+정렬 · 내부 선택’과 다르다. 고정 설정 비교 그림은 `fixed_settings_comparison.png`다.

‘이전 설정 고정’ 행은 기존 SVM의 센서·전처리·C·gamma를 유지하고 특징만 바꾼 통제 비교다. ‘내부 선택’ 행은 각 특징 방식 안에서 센서·전처리·C·gamma를 동일 범위로 다시 고른 것이다. 두 비교를 구분해 읽는다.

'''+ '\n'.join(sign_rows)+f'''

## 구현한 내용

- IMU 크기: 각 손의 가속도 벡터 크기와 자이로 벡터 크기의 평균·표준편차·최솟값·최댓값을 추가했다. 전체 구간과 세 시간 구간에 모두 추가하며 기존 축별 특징은 유지한다. 추가된 크기 특징 자체는 강체 축 회전에 불변이지만 전체 모델이 회전 불변이 됐다는 뜻은 아니다. 가속도에는 중력이 포함되며 물리적 자세 보정은 하지 않는다.
- 구간 정렬: 자이로 크기가 해당 동작 구간의 조용한 부분 통계와 고정 하한 10도/초를 넘고 이웃 샘플에서도 이어지는 구간을 찾는다. 최초·최후 움직임에 각각 250ms 여유를 붙인다. 최소 움직임 지속 200ms, 창 길이 1초·8샘플 이상, 재표본화 구간 최대 샘플 간격 250ms 조건을 통과하면 그 안에서만 32개 시간 점으로 재표본화한다.
- 움직임을 찾기 어렵거나 창이 너무 짧거나 간격이 크면 원래 전체 샘플과 구간을 유지한다. 가만히 있는 손을 0으로 만들거나 버리지 않는다. 원래 전체 구간 통계도 항상 유지하며 잘라낸 구간 비율·움직임 비율을 특징에 남긴다.
- 양손은 각각의 장치 시계에서 독립적으로 처리한다. 이 기능은 고정 측정이 끝난 뒤 구간을 정규화하는 처리다. 절대 시간 동기화, 정답 영상으로 검증된 동작 시작·끝 검출 또는 실시간 연속 인식을 구현했다는 뜻은 아니다.

## 실제 적용 현황

원시 전처리 기준 양손 창 2,340개 중 {good}개에 정렬·재표본화가 적용됐다. 나머지는 기존 창으로 되돌렸다. 이는 정렬 경계가 실제 수어와 정확히 일치했다는 정답률이 아니다. 전체 `alignment_audit.csv`는 전처리 세 종류를 따로 기록해 7,020행이며 새로운 측정이 7,020회라는 뜻이 아니다.

구간 정책의 임계값은 이번 시험 점수를 보고 조정하지 않았다. 현 버전은 자이로 변화가 작고 손가락 압력만 달라지는 동작을 놓칠 수 있어 기존 창 유지 규칙을 둔다. 모든 손에 정렬을 강제로 적용하지 않는다.

## 평가·검증

- 원본 고정 커밋은 `4d4d24a4d672c78d934bf120a59a193c2340637e`다. 파일별 SHA-256, 참가자·세션·회차·확인된 수어 별칭, XIAO/REAL 여부, 양손·IMU 유효성·시간·FSR 범위 검사를 유지했다.
- 이전 원시/필터/필터+FSR 정규화의 384개 기본 특징을 원본에서 다시 계산해 기존 특징표와 일치시켰다. 이전 SVM 473/1170 및 고정 KNN 416/1170의 회차별 예측도 재현했다.
- 사람별 바깥 5-fold와 각 학습 쪽의 사람별 3-fold는 이전 분할과 인덱스까지 동일하다. LOSO나 회차 무작위 분할을 사용하지 않았다. 한 참가자가 같은 시험의 학습·평가에 들어가지 않으며 모든 회차를 방법마다 한 번씩 시험한다.
- SVM 후보는 IMU/전체 센서, 원시/약한 필터/필터+손별 FSR 정규화, C 0.1·1·10·100, gamma scale·0.001·0.01·0.1을 비교한다. FSR만 쓰는 후보와 FSR 정규화 후 IMU만 쓰는 중복 후보는 제외한다. 총 320개 후보이며 네 특징 방식마다 같은 탐색 범위다.
- StandardScaler는 매 학습 부분에서만 적합한다. 설정 선택은 사람별·수어별 재현율을 고르게 반영한 평균으로 하고 동률 시 사전 후보 순서를 사용한다. 설정을 파일에 저장한 뒤 바깥 시험 예측을 수행한다.
- 가상 자료 테스트 결과는 실제 인식 정확도에 포함하지 않는다. 실제 회차별 예측은 `predictions.csv`, 사람별·수어별 결과는 각 CSV, 창 선택 및 폴백 사유는 `alignment_audit.csv`에 보존한다.

## 적용 범위

현재 기록에는 검증된 휴식 기준이 부족해 휴식값 빼기를 적용하지 않았다. 휴식 보정은 다음 측정에서 같은 사람·착용 세션의 양손 기준값을 받은 뒤 별도로 비교해야 한다.

이 자료는 과거에도 반복 분석한 1,170회다. 이번 시험 사람을 내부 설정 선택에 쓰지는 않았지만 새로운 외부 데이터는 아니다. 새 사람 또는 같은 팀원의 다른 날·재착용 세션에서의 정확도는 미측정이다. 기존 운영 모델을 덮어쓰거나 배포하지 않았다.

실행 방법은 `docs/imu-motion-study.md`, 후보/정책/입력 해시는 `plan.json`, 실제 설정은 `selected_configs.csv`, 검사 결과는 `verification.json`을 참조한다. 그래프는 모두 실제 예측에서 생성했으며 `accuracy_comparison.png/.pdf`, `selected_procedure_confusion.png`로 저장했다.
'''
    (out/'README.md').write_text(text,encoding='utf-8')
    base.dump(out/'report_hashes.json',{p.name:base.digest(p) for p in out.iterdir() if p.is_file()})
    print(out)


if __name__=='__main__':
    main()
