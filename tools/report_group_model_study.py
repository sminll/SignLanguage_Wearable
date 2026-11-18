"""Export real historical nested-CV results and a Korean report, never synthetic scores."""
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
import group_model_study as s

NAMES = {'fixed_existing_imu_knn':'기존 고정 IMU KNN', 'knn_whole':'KNN · 전체 통계',
 'svm_whole':'SVM · 전체 통계', 'knn_temporal':'KNN · 시간 구간 추가',
 'svm_temporal':'SVM · 시간 구간 추가', 'selected_whole':'학습 쪽에서 알고리즘 선택 · 전체 통계',
 'selected_any':'학습 쪽에서 알고리즘·특징 선택'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Fresh report directory required')
    source,args_out=args.results,args.out
    metrics=json.loads((source/'metrics.json').read_text(encoding='utf-8'))
    table=pd.read_csv(source/'summary.csv').set_index('method')
    predictions=pd.read_csv(source/'predictions.csv')
    for method,row in table.iterrows():
        part=predictions[predictions.method.eq(method)]
        assert len(part)==row.trials==1170
        assert part.file.nunique()==1170
        assert int(part.label.eq(part.prediction).sum())==row.correct
        assert not any(r.person in r.training_people.split('|') for r in part.itertuples())
    args_out.mkdir(parents=True)
    for name in ['summary.csv','per_sign.csv','per_person.csv','per_fold.csv','metrics.json',
                 'selected_configs.csv','split_manifest.json','source_manifest.json','evaluation_plan.json',
                 'verification.json','cohort_manifest.csv','predictions.csv']:
        shutil.copyfile(source/name,args_out/name)
    font=Path('C:/Windows/Fonts/malgun.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    order=list(NAMES)
    fig,ax=plt.subplots(figsize=(11,5))
    bars=ax.barh([NAMES[k] for k in order],table.loc[order,'accuracy']*100,color=['#909eae']+['#3b7fb8']*4+['#427c62','#285542'])
    for bar,k in zip(bars,order):
        r=table.loc[k]
        ax.text(bar.get_width()+.7,bar.get_y()+bar.get_height()/2,f'{int(r.correct)}/1170 · {r.accuracy:.2%}',va='center',fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0,100)
    ax.set_xlabel('참가자를 분리한 5-fold 전체 정답률 (%)')
    ax.set_title('XIAO 6개 수어 · 10명 · 동일한 사람별 시험 분할\n설정 선택은 각 학습 쪽의 3-fold에서만 수행')
    fig.tight_layout()
    fig.savefig(args_out/'accuracy_comparison.png',dpi=240)
    fig.savefig(args_out/'accuracy_comparison.pdf')
    plt.close(fig)
    result=next(r for r in metrics['results'] if r['method']=='selected_any')
    cm=np.array(result['confusion_matrix'])
    ratios=cm/cm.sum(axis=1,keepdims=True)*100
    fig,ax=plt.subplots(figsize=(8,7))
    im=ax.imshow(ratios,cmap='Blues',vmin=0,vmax=100)
    ax.set_xticks(range(6),metrics['labels'],rotation=30,ha='right')
    ax.set_yticks(range(6),metrics['labels'])
    for (i,j),value in np.ndenumerate(cm):
        ax.text(j,i,f'{value}회\n({ratios[i,j]:.1f}%)',ha='center',va='center',color='white' if ratios[i,j]>55 else '#173a5e')
    ax.set_xlabel('예측한 수어'); ax.set_ylabel('실제로 수행한 수어')
    ax.set_title(f"사람별 5-fold · 학습 쪽에서 알고리즘·특징 선택\n{result['correct']}/1170회 · {result['accuracy']:.2%}")
    fig.colorbar(im,ax=ax,label='실제 수어별 비율 (%)')
    fig.tight_layout(); fig.savefig(args_out/'selected_procedure_confusion.png',dpi=240); plt.close(fig)
    rows=['| 방법 | 정답 / 시험 횟수 | 전체 정확도 | 사람·수어를 고르게 반영한 점수 |',
          '|---|---:|---:|---:|']
    for method in order:
        r=table.loc[method]
        rows.append(f'| {NAMES[method]} | {int(r.correct)}/1170 | {r.accuracy:.2%} | {r.equal_person_class_recall:.2%} |')
    base=table.loc['fixed_existing_imu_knn']
    selected=table.loc['selected_any']
    svm_time=table.loc['svm_temporal']
    delta=int(selected.correct-base.correct)
    points=(selected.accuracy-base.accuracy)*100
    labels=pd.read_csv(source/'per_sign.csv')
    sign_rows=['| 수어 | 정답 / 시험 횟수 | 수어별 재현율 |','|---|---:|---:|']
    for r in labels[labels.method.eq('selected_any')].itertuples():
        sign_rows.append(f'| {r.label} | {r.sum}/{r.size} | {r.mean:.2%} |')
    text='''# KNN·SVM 및 시간 순서 특징: 사람별 5-fold 비교

10명·6개 수어·1,170회의 기존 XIAO 양손 데이터를 같은 시험 조건으로 비교했다. 사용한 수어는 괜찮다·아니다·무엇·마시다·화장실·싫다다. 참가자 ID는 숫자로 통일하고, 사용자가 확인한 별칭만 합쳤다.

## 결과

'''+ '\n'.join(rows)+f'''

미리 정한 ‘학습 쪽에서 알고리즘·특징 선택’ 절차는 기존 고정 IMU KNN 대비 정답 {delta:+d}회, 정확도 {points:+.2f}%p 차이다. 이 비교에서 기존 KNN도 동일한 5-fold로 새로 시험했다. 앞선 별도 subject4·10 시험의 41.43%와 직접 차이를 계산하지 않는다.

이번 보조 비교에서 SVM+시간 구간 특징은 {int(svm_time.correct)}/1170={svm_time.accuracy:.2%}로, 고정 IMU KNN 대비 {int(svm_time.correct-base.correct):+d}회, {(svm_time.accuracy-base.accuracy)*100:+.2f}%p였다. 이 방식은 다음 검증 후보로 남긴다. 자동 선택 절차가 오히려 낮아졌으므로, 탐색 후보를 늘리면 자동으로 좋아진다고 결론 내리지 않는다. 여러 방법 중 결과를 보고 이 후보를 고른 만큼 새로운 사람·새 세션에서 다시 확인해야 한다.

‘학습 쪽에서 선택’은 각 시험 사람을 본 뒤 가장 좋은 알고리즘을 고른다는 뜻이 아니다. 매번 시험할 두 사람을 빼고 남은 여덟 사람 안에서 설정을 정했다. 따라서 다섯 시험 묶음의 모델 설정이 서로 다를 수 있다. 표의 최고 점수 행을 사후에 골라 새 데이터의 성능이라고 주장하지 않는다.

아래 수어별 표와 기본 혼동행렬은 사전 지정한 자동 선택 절차의 결과다. SVM+시간 특징 결과와 구분한다. 모든 방법의 수어별·사람별 값은 CSV에 있다.

'''+ '\n'.join(sign_rows)+'''

## 실제로 한 일

1. 사람을 두 명씩 나눈 바깥 5-fold를 Seed 20261116으로 고정했다. 한 번에 여덟 명으로 개발하고 두 명을 시험했다. 10명 모두 학습에서 제외된 상태로 자기 기록을 한 번씩 시험받는다.
2. 각 개발 자료 안에서는 사람별 3-fold로 센서·전처리·모델 설정을 골랐다. 최종 시험 사람은 내부 설정 선택에도 들어가지 않는다. LOSO를 사용하거나 Fold 수를 점수가 높게 나오는 값으로 바꾸지 않았다.
3. 같은 모델 설정 탐색 범위를 두 특징 방식에 적용했다. 전체 통계는 양손 12개 센서의 평균·표준편차·최솟값·최댓값이다. 시간 특징은 이 통계에 앞·중간·뒤 세 구간의 같은 통계를 추가한다.
4. KNN은 이웃 수 1·3·5·7·11·15·21·31, 거리 L1/L2, 균등/거리 투표를 비교했다. SVM은 RBF 커널에서 C 0.1·1·10·100, gamma scale·0.001·0.01·0.1을 비교했다. IMU/FSR/전체 센서, 원시/약한 필터 및 필터+손별 FSR 정규화를 포함한 총 768개 후보를 각 내부 시험에서 비교했다.
5. StandardScaler는 매 학습 부분에서만 계산했다. 손별 FSR 정규화는 해당 동작 구간의 95분위 크기를 기준으로 하며, 휴식 기준 보정과 다르다. 검증된 휴식값이 없는 기존 기록에는 휴식값 빼기를 적용하지 않았다.
6. 기존 1,170회 명단의 Git 원본 파일 해시, XIAO/REAL 여부, 양손·IMU·시간·샘플 간격·FSR 범위 검사를 유지했다. 원본에서 다시 만든 원시/필터 전체 통계가 기존 특징표와 일치하는지 확인한 뒤 진행했다.

## 시간 특징의 범위

양손의 장치 시계 시작값은 서로 다를 수 있으므로 각 손에서 처음 받은 시각부터 마지막 시각까지를 독립적으로 삼등분했다. 각 구간 최소 두 샘플을 요구하며 누락 손을 0으로 채우지 않는다. 이 방식은 고정 측정 구간 안의 순서 정보를 추가하는 처리다. 양손 절대 시간 동기화, 동작 시작·끝 자동 검출이나 실시간 스트리밍 검증을 의미하지 않는다. 현재 시간 정보만으로 확보하지 못한 동기화를 임의로 추정하지 않았다.

기존 최단 측정 시간 기준은 3초 이상이며 최대 샘플 간격 500ms다. 명목상 5초 측정이더라도 실제 구간 길이는 파일에 따라 다를 수 있다. 이 실험에서 명단은 바꾸지 않았으며 5초 전체가 완벽히 기록됐다는 주장을 하지 않는다.

## 해석과 한계

동일한 과거 자료를 여러 번 분석한 탐색 결과다. 바깥 시험 자료를 이번 내부 설정 선택에는 쓰지 않았지만, 이 참가자들이 연구 전체에서 한 번도 분석되지 않았던 것은 아니다. 새 사람·새 측정 세션을 수집한 뒤 최종 후보를 고정해 시험해야 외부 정확도를 확인할 수 있다.

KNN과 SVM, 센서 조합, 전처리 후보까지 포함한 절차를 비교한 것이다. 시간 특징을 추가한 두 절차도 내부에서 서로 다른 센서나 설정을 고를 수 있으므로, 그 차이를 시간 특징 하나만의 순수한 효과로 해석하지 않는다. 평균 정확도만으로 개인차가 해결됐다고 판단하지 않고 `per_person.csv`, `per_sign.csv`와 혼동행렬을 함께 본다.

이번에는 평가만 수행했다. 기존 운영 모델을 덮어쓰거나 시험 결과를 보고 모델을 배포하지 않았다. `selected_any`는 내부 선택까지 포함한 평가 절차이며, 하나의 고정 설정으로 1,170회 전부 학습한 모델의 독립 시험 점수와 다르다.

## PR 방식으로 같은 결과가 나오는가

PR #8 코드를 고정 입력 커밋 `4d4d24a4d672c78d934bf120a59a193c2340637e`와 기본 설정으로 실행하면, 앞서 실제 재현한 양손 FSR 시험 386/454=85.02%와 일치해야 한다. 코드·라이브러리 버전도 맞추는 것이 좋다. `origin/training-data`는 새 업로드로 움직일 수 있으므로 고정 커밋을 사용한다.

PR의 85.02%는 원본 이름 24개 라벨과 같은 사람/착용 세션의 다른 회차 시험이다. 이 문서의 사람별 5-fold는 확정 별칭을 합친 6개 수어로 학습에 없는 사람을 시험하므로, 두 수치를 개선 전후로 비교하지 않는다.

## 근거 파일

- `summary.csv`, `per_person.csv`, `per_sign.csv`, `per_fold.csv`: 전체·사람·수어·묶음별 실제 정답 수.
- `predictions.csv`: 회차별 실제 라벨·예측·학습 사람·바깥 시험 묶음·선택 설정.
- `selected_configs.csv`, `split_manifest.json`, `evaluation_plan.json`: 실제 5개 바깥 시험 및 15개 내부 분할, 사전 후보 범위와 선택 규칙.
- `source_manifest.json`, `cohort_manifest.csv`, `verification.json`: 고정 입력 커밋·해시·명단·검증 결과.
- `accuracy_comparison.png/.pdf`, `selected_procedure_confusion.png`: 실제 위 예측에서 생성한 그림. 가상 테스트 점수는 포함하지 않는다.

실행 방법은 `docs/group-model-study.md`를 따른다. 이 결과는 별도 `feature/group-knn-svm-temporal` 브랜치에서 생성했다. GitHub push나 운영 모델 교체는 수행하지 않았다.
'''
    (args_out/'README.md').write_text(text,encoding='utf-8')
    s.dump(args_out/'report_hashes.json',{p.name:s.digest(p) for p in args_out.iterdir() if p.is_file()})
    print(args_out)


if __name__=='__main__':
    main()
