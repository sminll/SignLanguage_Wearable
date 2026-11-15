"""Explain descriptive sensor consistency separately from recognition accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from matplotlib import font_manager
import matplotlib.pyplot as plt

try:
    from .report_six_sign_knn import draw_table, md_table
except ImportError:
    from report_six_sign_knn import draw_table, md_table


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    args=parser.parse_args()
    root=args.results.resolve()
    out=root/'공유자료'
    if out.exists():
        raise FileExistsError('Use a fresh share directory')
    meta=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    result=pd.DataFrame(meta['results'])
    pairs=pd.read_csv(next(root.glob('사람쌍*.csv')))
    assert meta['raw_files_verified']==1603 and meta['person_sign_templates']==82
    for row in result.to_dict('records'):
        for sensor in ['FSR','IMU']:
            g=pairs[pairs.label.eq(row['label']) & pairs.sensor.eq(sensor)]
            assert len(g)==row['pairs']
            assert np.isclose(g.scaled_waveform_distance.median(),row[sensor+'_person_distance'])
    for path,expected in meta['input_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
    result['FSR_rank']=result.FSR_person_distance.rank(method='min').astype(int)
    result['IMU_rank']=result.IMU_person_distance.rank(method='min').astype(int)
    result=result.sort_values(['FSR_rank','IMU_rank'])
    table_rows=[[r.label,r.people,r.trials,int(r.FSR_rank),int(r.IMU_rank),r.left_zero,r.right_zero]
                for r in result.itertuples()]
    out.mkdir()
    available={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in ['Malgun Gothic','Noto Sans CJK KR','NanumGothic','AppleGothic'] if f in available),None)
    if not font:
        raise ValueError('Korean font required')
    plt.rcParams.update({'font.family':font,'axes.unicode_minus':False,'svg.fonttype':'none'})
    draw_table('직접 센서 비교 — 사람 사이의 값 차이가 작은 순서',
               ['수어','참가자','회차','FSR 순위','IMU 순위','왼손 전체0','오른손 전체0'],table_rows,
               '각 순위는 9종 안에서 상대 비교, 1위가 사람 간 평균 파형 차이가 가장 작음\n'
               '정확도나 합격 점수 아님 · 사람마다 반복회차를 평균하여 동일 가중 비교 · 날짜/착용 조건 차이 포함',
               out/'센서값_사람간차이_순위')
    result.to_csv(out/'센서일관성_전체표.csv',index=False,encoding='utf-8-sig')
    checks=[]
    for label in ['삼키다','상처','아니다']:
        r=result[result.label.eq(label)].iloc[0]
        checks.append([label,int(r.FSR_rank),int(r.IMU_rank)])
    # An offset/residual decomposition is descriptive, not an added selector.
    z=np.load(root/'person_templates.npz')
    scales=pd.read_csv(root/'channel_scales.csv',index_col=0).to_numpy(float)
    waves=z['waveforms']/scales[None,:,None,:]
    offset_rows=[]
    for label in sorted(set(z['labels'])):
        ids=np.flatnonzero(z['labels']==label)
        for sensor,sl in [('FSR',slice(0,6)),('IMU',slice(6,12))]:
            fractions=[]
            for i,ia in enumerate(ids):
                for ib in ids[i+1:]:
                    diff=waves[ia,:,:,sl]-waves[ib,:,:,sl]
                    energy=float(np.mean(diff*diff))
                    offset=float(np.mean(diff.mean(axis=1)**2))
                    if energy>1e-12:
                        fractions.append(offset/energy)
            offset_rows.append(dict(label=label,sensor=sensor,
                                    median_level_offset_fraction=float(np.median(fractions))))
    pd.DataFrame(offset_rows).to_csv(out/'평균값차이_기여도.csv',index=False,encoding='utf-8-sig')
    report='''# 2026-11-15 사람 간 센서값 일관성 직접 비교

이번에는 KNN 정답률로 수어를 고르지 않고 원본 FSR·IMU 파형을 직접 비교했다.
9종 수어,1,603회,82개 사람×수어 평균 파형을 사용했다. 기존3종559회와 새6종양손1,044회이며
기존 손 누락16회는 이전 분석과 동일하게 미포함하고 원본을 보존했다. 새로 제외한 자료는 없다.

## 쉬운 결과표

각 센서의 순위는 사람끼리 값 차이가 작은 순서다.1위가 가장 비슷하다는 상대 순위이며,
1위라고 정확도가 가장 높거나 사용할 수 있다고 보장하지 않는다. 두 센서의 순위는 각각 계산했다.

'''+md_table(['수어','참가자','평가회차','FSR 순위','IMU 순위','왼손 FSR6개 전체0','오른손 FSR6개 전체0'],table_rows)+'''

## 먼저 골랐던3종 확인

'''+md_table(['수어','FSR 순위','IMU 순위'],checks)+'''

- 삼키다: FSR3위·IMU4위로 두 센서를 함께 볼 때 비교적 중간 이상이다.
- 상처: FSR1위이나 IMU9위다. 압력값이 비슷하다는 점만으로 움직임도 비슷하다고 할 수 없다.
- 아니다: FSR5위·IMU8위로 직접 센서값 일관성에서는 상위권이라고 보기 어렵다.
- FSR차이 상위3종은 상처·교통사고·삼키다이고 IMU차이 상위3종은 화장실·무엇·싫다다.
  두 상위3종 집합에 공통으로 포함되는 수어는 없다.

따라서 기존에 고른 삼키다·아니다·상처는 '정답률로 고른 시연 후보'이지,
'두 센서 모두 사람 간 값이 일관된3종'으로 검증된 것은 아니다.
현재로서는 삼키다를 우선 확인할 후보로 두고, 상처의IMU/아니다의사람별 차이를 확인해야 한다.
단순 순위만으로 교통사고나화장실로 바꾸는 것도 맞지 않는다. 그 수어들의 기존 분류 정답률은 낮기 때문이다.

## 값의 일관성과 동작 구분을 함께 본 참고 지표

수어별 사람 간 평균 파형 거리(중앙값)를 가장 가까운 다른 수어의 전체평균 파형 거리로 나눈 참고비율도 계산했다.
1보다 작으면 그 수어의 사람 간 차이가 가장 가까운 다른 수어 평균과의 차이보다 작다는 뜻이다.
이번에는9종 모두FSR·IMU에서1미만을 동시에 만족하지 못했다. 이는 임의로 정한 탐색 기준이며,
수어가 쓸모없거나 모든 분류기가 실패한다는 뜻은 아니다. 전체평균 하나가 여러 사람의 다양한 패턴을
대표하지 못할 수 있으며, KNN은 하나의 평균 대신 각각의 예시를 이용한다.
이 비율이나 직접 파형 순위를 인식 정확도 백분율로 바꾸어 쓰지 않는다.

## FSR0 해석

양손FSR12개가 회차 내내 모두0인 파일은 이번1,603회에0개다. 한쪽 손의6개가 전체0인 경우는 표처럼 있다.
한 손만 사용하는 동작의 반대 손일 수 있어0만으로 불량처리하지 않는다. 센서별로 일부0인 채널도 별도 기록했다.
원본의0값을 센서값이 전송되지 않은 것으로 바꾸어 해석하지 않는다. 실제 사용하는 손/센서가 기록으로
확인되지 않으면 사용손FSR가 정상인지의 최종 판정은 보류한다.

## 실제 계산 방법과 한계

1. 각파일 원본을 고정Git커밋에서 읽어 SHA256확인,96통계특징을 재계산하여 이전값과 일치 확인.
2. 각손은 자기 기록 시작~끝의 상대진행률로50시점 보간. 서로 다른 보드의 부팅시계를 직접 맞추지 않음.
3. 사람·수어별 반복파형을 평균하여 한 파형으로 만들고, 각사람에 같은 가중치를 줌.
4.82개 평균파형 전체에서 손·채널별 공통 표준편차로 단위를 맞춘 뒤 사람 쌍의 RMS거리를 계산.
  이 표준화는 설명통계용이며 분류모델을 학습하거나 검증하는 과정이 아니다.
5.FSR12채널과IMU12축을 각각 비교하며 수어별 사람쌍 거리 중앙값을 사용. 원본FSR0는 보간 전 확인.
6.시간별 평균을 제거한 파형 형태의 cosine도 따로 저장. 평탄한 신호에는 완벽한 유사도를 주지 않음.
  사람마다 동작시점/속도가 달라 위상이 어긋날 수 있어 낮은형태유사도만으로 잘못된 수어로 판정하지 않음.

같은 수어를 다른 사람들이 수행한 값이 어느 정도 달랐는지 직접 보는 탐색적 분석이다.
센서값 차이가 착용 방향·압박·해부학적 차이·수어차이 중 무엇 때문인지는 영상/착용기록 없이 확정하지 못한다.
수어별 날짜와 세션이 달라 같은조건의 반복 실험을 대체하지 않는다. 작은파형거리는 평탄한 신호일 수도 있어
분류 정확도와 함께 봐야 한다. rawFSR/IMU와 기존모델을 수정하거나 삭제하지 않았다.

## 파일과 재현

`회차별_FSR상태.csv`, `사람쌍별_파형비교.csv`, `수어별_센서일관성.csv`,
`person_templates.npz`, `channel_scales.csv`, `comparison_plan.json`, `summary.json`에 근거 보존.
별도 `평균값차이_기여도.csv`는 사람쌍 파형 차이를 시간평균값 차이와 나머지 변화 차이로 분해한 설명통계이다.
후보 순위나 인식 점수의 추가선택 기준으로 쓰지 않았다.

```powershell
python -B tools/compare_sign_sensor_consistency.py --results-root data/knn_results --repo . --out data/knn_results/sensor_consistency_rerun
python -B tools/report_sensor_consistency.py --results data/knn_results/sensor_consistency_rerun
python -B -m unittest discover -s tests -p test_sensor_consistency.py -v
```

4개 합성fixture 테스트로 동일인파형거리0, 평탄신호의 가짜유사도 방지, 단위변경불변성,
독립시계/0FSR/손누락 처리를 확인했다. 합성자료를 실제 분석에 섞지 않았다.
'''
    (root/'report.md').write_text(report,encoding='utf-8')
    (out/'센서일관성_보고서.md').write_text(report,encoding='utf-8')
    shutil.copy2(Path(__file__),root/Path(__file__).name)
    (out/'파일_SHA256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                                  for p in out.iterdir()},ensure_ascii=False,indent=2),encoding='utf-8')
    print(result[['label','FSR_rank','IMU_rank','left_zero','right_zero']].to_string(index=False))
    print('Saved verified direct comparison report:',out)


if __name__ == '__main__':
    main()
