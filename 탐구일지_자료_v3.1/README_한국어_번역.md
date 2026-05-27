# EmoNet v3.1

v3.1은 다음 아이디어를 검증하기 위한 연구 브랜치이다.

> trace는 감정에 대한 부가 설명이 아니라, 감정 상태 표현 그 자체이다.

v4가 앱과 평가 구현 브랜치라면, v3.1은 EmoNet의 과학적 질문에 집중한다. 즉 neural/appraisal trace가 안정적이고 구조화된 감정 상태 공간을 형성하는지 확인하는 것이다.

## 핵심 가설

이전 파이프라인은 trace를 설명 자료 또는 응답 생성을 위한 조건 정보로 사용했다.

```text
stimulus -> episode trace -> response prompt -> generated response
```

v3.1에서는 trace를 감정 자체로 다룬다.

```text
stimulus -> trace dynamics -> emotion-state representation -> appraisal/action constraints -> response
```

따라서 증명 목표가 달라진다. 단순히 생성된 응답이 더 자연스럽거나 좋은지를 묻는 대신, v3.1은 trace 공간 자체가 감정과 비슷한 구조를 가지는지 묻는다.

- 비슷한 감정 상황은 가까운 trace를 만든다.
- 서로 다른 appraisal/action 패턴은 trace 공간에서 분리된다.
- trace를 바꾸면 감정 해석도 바뀐다.
- trace를 제거하면 appraisal fidelity와 affect preservation이 약해진다.

## 디렉터리 구조

```text
v3.1/
  README.md
  docs/
    TRACE_AS_EMOTION_DESIGN.md
    EXPERIMENT_ROADMAP.md
  scripts/
    trace_emotion_probe.py
  outputs/
    .gitkeep
```

## 첫 번째 실험

첫 번째 probe는 의도적으로 가볍게 설계되었다. targeted record CSV를 읽고, trace field를 간단한 범주형/수치형 혼합 표현으로 바꾼 뒤, trace 공간에서 가까운 이웃들이 같은 감정 속성을 공유하는지 보고한다.

예시 입력:

```text
../v4/outputs/experiments/superiority_targeted_v1/targeted_records.csv
```

예시 출력:

```text
outputs/trace_emotion_probe_summary.json
```

## 이 브랜치가 필요한 이유

v4의 targeted superiority 결과는 trace 안에 유용한 감정 정보가 들어 있을 가능성을 보여준다. 그러나 응답 품질만으로는 간접 증거에 그친다. v3.1은 그 주장을 representation level에서 검증 가능한 문제로 바꾸기 위해 존재한다.
