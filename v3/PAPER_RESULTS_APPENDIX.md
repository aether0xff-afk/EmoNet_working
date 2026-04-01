# Paper Results Appendix

## 현재 로컬 산출물 기준 핵심 수치

| Item | Value |
| --- | ---: |
| `out_z_training.csv` rows | 51,628 |
| unique labels in `out_z_training.csv` | 60 |
| `llm_subset.csv` rows | 500 |
| `llm_subset_labeled_200_ollama.csv` rows | 200 |
| keep rows in `llm_subset_labeled_200_ollama.csv` | 190 |
| keep rate | 95.0% |
| mean consistency L1 | 0.0998 |
| median consistency L1 | 0.0938 |

## 현재 스타일 분포 편향

`llm_subset_labeled_200_ollama.csv`의 keep 샘플 평균 기준:

| Axis | Mean |
| --- | ---: |
| warmth | 0.7776 |
| politeness | 0.7789 |
| cooperativeness | 0.9197 |
| calmness | 0.9184 |
| softness | 0.9605 |
| positivity | 0.9092 |
| seriousness | 0.6961 |
| dominance | 0.1066 |
| tension | 0.0961 |
| sharpness | 0.0382 |

이 값들은 현재 스타일 공간이 매우 부드럽고 협조적이며 차분한 응답으로 치우쳐 있음을 보여준다.

## z->s 회귀기 간이 검증

`llm_subset_labeled_200_ollama.csv`의 keep 샘플 190개를 사용해 5개 seed hold-out split으로 평가했다. 각 split의 검증셋 크기는 19이다.

| Seed | Decoder MAE | Mean Baseline MAE | Gain |
| --- | ---: | ---: | ---: |
| 7 | 0.1406 | 0.1324 | -0.0082 |
| 13 | 0.1344 | 0.1301 | -0.0042 |
| 21 | 0.1469 | 0.1411 | -0.0057 |
| 42 | 0.1476 | 0.1438 | -0.0038 |
| 84 | 0.1414 | 0.1378 | -0.0036 |
| Mean | 0.1422 | 0.1371 | -0.0051 |

현재 선형 `z -> s` 회귀기는 단순 mean baseline보다 아직 낮지 않은 MAE를 보인다. 즉, 스타일 공간이 편향돼 있어 예측기가 대부분의 축에서 평균값만 따라가도 일정 수준 성능이 나오는 상황일 가능성이 있다.

## 해석 메모

- 라벨링 consistency는 높다.
- 그러나 스타일 공간 자체가 좁고 한 방향으로 몰려 있다.
- 따라서 `z -> s` 회귀기의 절대 MAE만 볼 것이 아니라, baseline 대비 개선 여부를 같이 봐야 한다.
- 지금 상태에서는 모델 성능보다 스타일 축 설계와 데이터 구성의 편향이 더 큰 병목으로 보인다.
