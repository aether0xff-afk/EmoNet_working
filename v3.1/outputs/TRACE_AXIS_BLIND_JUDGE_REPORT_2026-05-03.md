# Trace Axis-Only Blind Judge Report

작성일: 2026-05-03

## 1. 왜 다시 평가했나

이전 pairwise LLM judge는 `while remaining natural` 같은 조건이 포함되어 있어, 판정기가 감정축이 아니라 답변의 자연스러움, 친절함, 상담 품질을 고를 위험이 있었다.

v3.1의 목표는 좋은 답변 생성이 아니라 다음 명제다.

```text
neural trace가 감정 상태 표현 후보이고,
trace 조작이 감정축 방향을 움직이는가?
```

따라서 새 judge는 감정축만 보도록 제한했다.

## 2. 새 판정 방식

새 스크립트:

- `scripts/judge_trace_axis_blind.py`

핵심 변경:

- `trace_full`, `ablation`, `perturbation` 조건명을 judge prompt에서 숨김
- A/B 순서를 deterministic hash로 섞음
- helpfulness, warmth, politeness, fluency, empathy, overall quality 판정 금지
- 오직 지정된 감정축만 판정
- 동일 응답끼리 비교하는 null pair 추가
- null pair에서는 `tie`가 나와야 함

판정 축:

| 기존 이름 | 한국어 이름 |
|---|---|
| `target` | 감정이 향하는 대상 |
| `social_orientation` | 사회적 방향 |
| `control_state` | 통제감 상태 |
| `action_tendency_class` | 행동 경향 |

## 3. 실행

명령:

```powershell
cd .\v3.1
$env:ANTHROPIC_API_KEY = "<local only>"
python .\scripts\judge_trace_axis_blind.py --model claude-haiku-4-5-20251001 --max-output-tokens 800 --output outputs\trace_axis_blind_judgments_claude45haiku_dry3.csv --summary outputs\trace_axis_blind_judgments_claude45haiku_dry3_summary.json
```

실행상 수정:

- 260/500 output tokens에서는 일부 JSON이 잘리거나 인용부호 때문에 파싱 오류가 났다.
- prompt에 `직접 인용 대신 짧은 의역 근거`를 요구하고 `max-output-tokens=800`으로 올려 36/36 정상 파싱을 달성했다.

## 4. 결과

전체:

| Scope | n | success | success rate | tie rate |
|---|---:|---:|---:|---:|
| Overall | 36 | 26 | 0.722222 | 0.444444 |
| Ablation axis original | 12 | 4 | 0.333333 | 0.250000 |
| Perturbation axis shift | 12 | 10 | 0.833333 | 0.083333 |
| Null same response | 12 | 12 | 1.000000 | 1.000000 |

축별:

| Axis | n | success rate | tie rate |
|---|---:|---:|---:|
| 감정이 향하는 대상 | 9 | 0.666667 | 0.444444 |
| 사회적 방향 | 9 | 0.777778 | 0.444444 |
| 통제감 상태 | 9 | 0.777778 | 0.333333 |
| 행동 경향 | 9 | 0.666667 | 0.555556 |

## 5. 해석

이 결과는 이전 LLM judge 우려를 상당 부분 줄여준다.

```text
좋은 답변/자연스러운 답변을 고르지 말라고 해도,
trace perturbation은 여전히 10/12에서 조작된 감정축 방향으로 판정되었다.
```

또한 null pair가 12/12 모두 tie로 나왔으므로, judge가 동일 응답에서도 억지로 A/B를 고르는 강한 편향은 보이지 않았다.

하지만 ablation은 여전히 약하다.

```text
정보 제거 조건은 4/12만 성공했다.
즉 현재 ablation은 해당 감정축 정보를 지우는 데 충분히 강하지 않다.
```

## 6. 결론

v3.1 causal evidence의 현재 상태:

| Evidence | 판정 |
|---|---|
| 방향 교란 | 강함. axis-only blind judge에서도 10/12 |
| 무효 비교 | 정상. 동일 응답은 12/12 tie |
| 정보 제거 | 약함. 4/12 |

따라서 논문에서 가능한 주장은 다음이다.

```text
Trace perturbation changes response-level emotion-axis interpretation,
even under an axis-only blind judge that ignores response quality.
```

아직 하면 안 되는 주장은 다음이다.

```text
Trace ablation proves causal necessity of each emotion axis.
```

## 7. 다음 완성 작업

완성본으로 가려면 ablation을 단순 제거가 아니라 더 강한 중립화로 바꿔야 한다.

권장:

1. 해당 axis field 제거
2. preserve/avoid/action 문장 안의 같은 축 단서도 제거
3. 중립 문장으로 대체
4. 원본과 중립화 응답을 axis-only blind judge로 재평가
5. ablation success를 최소 0.60 이상으로 올리는지 확인
