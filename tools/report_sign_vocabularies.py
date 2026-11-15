"""Explain original-three / four / fifth-sign comparisons without claiming external accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

try:
    from . import compare_sign_vocabularies as vocab
    from .report_six_sign_knn import draw_table,md_table,pct,save_figure
except ImportError:
    import compare_sign_vocabularies as vocab
    from report_six_sign_knn import draw_table,md_table,pct,save_figure

NOTE='같은 자료를 반복 비교한 탐색적 내부 평가 · 수어별 날짜/착용 조건 차이 포함 · 새 외부 검증 아님'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    args=parser.parse_args()
    root=args.results.resolve();out=root/'공유자료'
    if out.exists():
        raise FileExistsError('Share folder already exists')
    meta=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    rows=meta['results'];best=meta['chosen_five']
    highest=sorted([r for r in rows if 'extra' in r],key=lambda r:(-r['accuracy'],r['name']))[0]
    if vocab.choose_five([r for r in rows if 'extra' in r])['name']!=best['name']:
        raise ValueError('Selection differs from predeclared review rule')
    for row in rows:
        predictions=pd.read_csv(root/row['name']/'outer_predictions.csv')
        cohort=pd.read_csv(root/row['name']/'cohort_manifest.csv')
        prior=json.loads((root/row['name']/'evaluation.json').read_text(encoding='utf-8'))
        for method,part in predictions.groupby('method'):
            part=part.copy();part['correct']=part.label.eq(part.prediction)
            if part.file.duplicated().any() or set(part.file)!=set(cohort.file):
                raise ValueError('Predictions do not cover each source trial once')
            if any(r.person in r.train_people.split('|') for r in part.itertuples()):
                raise ValueError('Outer-person leakage')
            actual=vocab.tri.summary(part,row['labels'])
            if json.dumps(actual,sort_keys=True)!=json.dumps(prior[method],sort_keys=True):
                raise ValueError('Saved metrics differ from predictions')
        assert np.isclose(row['accuracy'],prior['nested_selection']['accuracy'])
    available={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in ['Malgun Gothic','Noto Sans CJK KR','NanumGothic','AppleGothic'] if f in available),None)
    if font is None:
        raise ValueError('Korean font required')
    plt.rcParams.update({'font.family':font,'axes.unicode_minus':False,'svg.fonttype':'none'})
    out.mkdir()
    table_rows=[]
    for r in rows:
        name='기존3종' if r['name']=='original3' else '+아니다 4종' if r['name']=='plus_denial4' else '5종: +'+r['extra']
        table_rows.append([name,f"{r['correct']}/{r['trials']}",pct(r['accuracy']),pct(r['minimum_sign_recall'])])
    draw_table('기존3종 + 아니다, 그리고 다섯 번째 수어 비교',
               ['구성','정답/평가','전체 정확도','최저 수어 정답률'],table_rows,
               '기존3종=교통사고/삼키다/상처 · 5종은 이3개+아니다+표의 추가 수어\n'+NOTE,
               out/'01_3종_4종_5종_비교')
    sign_rows=[]
    for r in rows:
        for sign in r['per_sign']:
            sign_rows.append(dict(vocabulary=r['name'],labels='/'.join(r['labels']),**sign))
    pd.DataFrame(sign_rows).to_csv(out/'전체구성_수어별정답률.csv',index=False,encoding='utf-8-sig')
    details=[]
    person_tables=[]
    targets=[('4종',next(r for r in rows if r['name']=='plus_denial4')),('5종 균형 후보',best)]
    if highest['name']!=best['name']:
        targets.append(('5종 전체정확도 최고',highest))
    for title,spec in targets:
        folder=root/spec['name']
        evaluation=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))['nested_selection']
        detail=[[s['label'],s['trials'],s['correct'],pct(s['recall']),pct(s['precision'])] for s in spec['per_sign']]
        details.append((title,spec,detail))
        draw_table(title+' — 수어별 결과',
                   ['실제 수어','평가 회차','정답','정답률','해당 판정의 정답 비율'],detail,
                   '한 사람을 학습과 설정 선택에서 제외하는 평가를10명에게 반복\n'+NOTE,
                   out/(title+'_수어별결과'))
        person_rows=[[p['person'],'/'.join(p['labels']),p['trials'],p['correct'],pct(p['accuracy'])]
                      for p in evaluation['people']]
        person_tables.append((title,person_rows))
        pd.DataFrame(evaluation['people']).to_csv(out/(title+'_참가자별결과.csv'),index=False,encoding='utf-8-sig')
        cm=np.array(evaluation['confusion_matrix']);rates=100*cm/cm.sum(axis=1,keepdims=True)
        n=len(spec['labels'])
        fig,ax=plt.subplots(figsize=(11,9.6))
        pic=ax.imshow(rates,cmap='Blues',vmin=0,vmax=100)
        ax.set_xticks(range(n),spec['labels'],fontsize=13)
        ax.set_yticks(range(n),spec['labels'],fontsize=13)
        ax.set_xlabel('예측한 수어',fontsize=15);ax.set_ylabel('실제 수행한 수어',fontsize=15)
        for i in range(n):
            for j in range(n):
                ax.text(j,i,f'{cm[i,j]}회\n({rates[i,j]:.1f}%)',ha='center',va='center',fontsize=12,
                        color='white' if rates[i,j]>55 else '#123b60')
        fig.colorbar(pic,ax=ax,fraction=.046,pad=.04,label='실제 수어별 비율 (%)')
        fig.suptitle(title+' — 참가자 제외 평가',fontsize=20,fontweight='bold')
        ax.set_title(f"{spec['correct']}/{spec['trials']}회 정답 · {pct(spec['accuracy'])}",fontsize=15,pad=18)
        fig.text(.5,.02,NOTE,ha='center',fontsize=10)
        fig.subplots_adjust(top=.87,bottom=.14)
        save_figure(fig,out/(title+'_혼동행렬'))
        pd.DataFrame(cm,index=spec['labels'],columns=spec['labels']).to_csv(
            out/(title+'_혼동행렬.csv'),encoding='utf-8-sig')
    base_four=next(r for r in rows if r['name']=='plus_denial4')
    change_rows=[[r['extra'],f"{r['base_four_correct']}/{r['base_four_trials']}",pct(r['base_four_accuracy']),
                  pct(r['extra_recall'])] for r in rows if 'extra' in r]
    report='''# 2026-11-15 기존3종+아니다4종 / 5종 후보 비교

기존3종(교통사고·삼키다·상처)에 아니다를 추가하는4종과, 그4종을 유지하고 나머지5개 수어를 하나씩 더하는5종을 비교했다.
모든9종에서 가능한5종 조합126개를 전부 비교한 것은 아니다.
각 구성은 하나의 실제 분류기로 학습·예측했으며 기존 수어별 정확도를 더하거나 평균낸 것이 아니다.

**'''+NOTE+'''**

## 전체 비교

'''+md_table(['구성','정답/평가','전체 정확도','최저 수어 정답률'],table_rows)+f'''

선택한5종 후보: **{'/'.join(best['labels'])}**.
미리 정한 기준은 가장 낮은 수어의 정답률이 높은 구성, 다음macro recall, 다음전체정확도, 마지막고정이름순서이다.
이 기준은 특정수어를 거의못맞혀도 전체평균만 높은 구성을 피하기 위한 내부검토 기준이다.
이 같은 바깥 예측을 보고 수어 구성을 골랐으므로 선택한5종 점수를 독립적인 확증 결과로 표시하지 않는다.
원한다면 가장 높은 전체점수와 선택기준의 결과가 다를 수 있으며 이 차이를 숨기지 않는다.
실제 전체정확도가 가장 높은5종은 **{'/'.join(highest['labels'])}**이며
{highest['correct']}/{highest['trials']}({pct(highest['accuracy'])})다.
가장 낮은수어 정답률은 {pct(highest['minimum_sign_recall'])}다.

## 평가 방법과 자료

- 이전 raw검사에서 해시와96통계특징을 검증한1,603회 특징표를 재사용. 이전검사 이후 파일해시 일치 확인.
- 원래 사용하던 초기제외/손누락16회 기준 유지. 낮은성능이나0FSR로 새로 제외하지 않음.
- 10명 중 한 사람의 모든 날짜·회차를 빼고 나머지 사람만으로 학습.
- 안쪽에서도 사람을 나눠36설정(센서3종 × K3/5/7 × 거리2 × 가중치2)을 선택.
- 일부수어만 있는 사람도 학습과 바깥평가에 포함. 안쪽선택점수는 해당구성의 모든수어가 있는 학습사람에게서 계산.
- StandardScaler는 각학습split에서만 fit. 모든유효회차를 방법별로 한 번씩 평가.
- 주 결과는 센서입력까지 안쪽에서 고르는 `nested_selection`이다.
- 기존3종도 이번과 같은 선택절차로 다시 평가했다. 과거73.9%는 고정된IMU설정의 사후내부평가여서 동일한 절차가 아니다.
- 서로다른수어/회차수가 있는 분류문제끼리 비교하는 것이며 동일시험집합에서 단일알고리즘의 향상이라고 표시하지 않는다.

## 4종을 유지한 채 다섯 번째 수어를 넣었을 때

4종의739회가 모든5종에서도 동일하게 남았는지 파일단위로 확인했다.
4종만 분류했을 때는 {base_four['correct']}/739({pct(base_four['accuracy'])})이다.

'''+md_table(['추가 수어','기존4종 회차 정답','기존4종 정답률','추가수어 정답률'],change_rows)+'\n\n'
    for title,spec,detail in details:
        report+='## '+title+' 수어별 결과\n\n'+md_table(
            ['실제 수어','평가회차','정답','정답률','해당 판정의 정답 비율'],detail)+'\n\n'
    for title,people in person_tables:
        report+='## '+title+' 사람별 결과\n\n'+md_table(
            ['참가자','실제로 측정한 수어','평가회차','정답','정확도'],people)+'\n\n'
    report+='''## 해석 한계와 후보 파일

기존결과를 보고 수어구성을 여러번 시험한 탐색이다. 새참가자 외부평가가 아니며 수어별날짜/착용조건차이가 남아있다.
5종 후보를 고른 뒤 같은자료를 더 평가해 독립시험점수라고 다시 부르지 않는다. 새측정없는 현재상황에서 제출시 이범위를 명시한다.
FSR/IMU의 센서값 일관성 순위와 분류정확도는 다른지표이며 이번 높은점수가 사람간센서값이 일정하다는 증거는 아니다.

각하위폴더 `candidate_model.joblib`은 평가후 해당구성의 전체자료로 별도학습했다.
기존모델을 덮어쓰거나 서버/앱에 자동적용하지 않았다. 최종후보파일의 새외부정확도는 모두미측정이다.
각폴더에 원본해시/회차목록,예측CSV,바깥fold,안쪽선택표,전체자료설정을 보존했다.
보고서의 모든숫자는 저장된회차별예측으로 재계산하여 확인했다. 합성테스트숫자는 실험에 포함하지 않았다.

## 재현

```powershell
python -B tools/compare_sign_vocabularies.py --results-root data/knn_results --out data/knn_results/vocabulary_comparison_rerun
python -B tools/report_sign_vocabularies.py --results data/knn_results/vocabulary_comparison_rerun
```

이전원본파형직접비교의summary와회차목록이 필요하다. 출력폴더는 새이름을 지정한다.
'''
    (root/'report.md').write_text(report,encoding='utf-8')
    (out/'비교_보고서.md').write_text(report,encoding='utf-8')
    pd.read_csv(root/'vocabulary_comparison.csv').to_csv(out/'전체_비교.csv',index=False,encoding='utf-8-sig')
    team=(f"기존3종에 아니다를 붙인4종은 {base_four['correct']}/{base_four['trials']}({pct(base_four['accuracy'])})였습니다.\n"
          f"기존4종을 유지한5종 중 전체정확도 최고는 {'/'.join(highest['labels'])}, "
          f"{highest['correct']}/{highest['trials']}({pct(highest['accuracy'])})였습니다.\n"
          f"그4종을 유지한5종 후보 중에는 {'/'.join(best['labels'])}가 최저수어정답률 기준으로 선택됐고, "
          f"전체 {best['correct']}/{best['trials']}({pct(best['accuracy'])})였습니다.\n"
          '수어별: '+' / '.join(f"{s['label']} {pct(s['recall'])}" for s in best['per_sign'])+'\n'
          '같은기존자료에서 후보구성을 비교한 내부평가이며, 수어별측정날짜 차이가 남아있습니다. '
          '센서값이 모든사람에게 일관되거나 새사람에게 같은정확도가 나온다는 뜻은 아닙니다.\n')
    (out/'팀원_공유문구.txt').write_text(team,encoding='utf-8')
    shutil.copy2(Path(__file__),root/Path(__file__).name)
    vocab.base.dump_json(out/'파일_SHA256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()})
    print(team)
    print('Validated vocabulary comparison report saved:',out)


if __name__=='__main__':
    main()
