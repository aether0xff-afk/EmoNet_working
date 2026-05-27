# Trace-As-Emotion 설계

## 1. 입장

v3.1의 작업 가설은 다음과 같다.

> 감정은 분노, 수치심, 죄책감, 슬픔 같은 라벨만이 아니다. 감정은 시간에 따라 형성되는 appraisal, target, control, social orientation, affect intensity, action tendency의 구조화된 trace이다.

이 관점에서 `trace`는 보조 정보가 아니다. trace는 내부 감정 상태 표현이다.

## 2. v4와의 차이

v4는 현재 유용한 공학적 사실을 증명한다.

```text
episode trace는 targeted episode-sensitive input에서 생성 품질을 돕는다
```

v3.1은 더 깊은 연구 질문을 묻는다.

```text
trace space 자체가 emotion space처럼 작동하는가?
```

이 구분은 중요하다.

| 버전 | trace의 주된 역할 | 증명 목표 |
|---|---|---|
| v4 | 프롬프트 조건 정보 | targeted response 개선 |
| v3.1 | 감정 상태 표현 | 구조화되고 안정적이며 조작 가능한 emotion space |

## 3. 감정 상태 구성 요소

현재 trace field는 감정 상태 축으로 해석할 수 있다.

| Field | 감정 상태에서의 역할 |
|---|---|
| `valence` | pleasant/unpleasant 방향 |
| `arousal` | 활성화 또는 강도 |
| `target` | 감정이 향하는 자기, 타인, 상황, 미지 대상 |
| `control_state` | 통제 가능성, 무력감, 행위 주체감 |
| `social_orientation` | episode의 사회적 방향 |
| `preserve` | 응답에 남아 있어야 하는 정서적 내용 |
| `avoid` | 감정을 왜곡할 수 있어 피해야 하는 응답 패턴 |
| `action_tendency` | 충동 또는 행동 방향 |
| `episode_label` | 거친 episode 분류이며, 감정 전체는 아님 |

중요한 주장은 감정이 단일 라벨이 아니라, 이 축들의 배치와 시간적 궤적이라는 점이다.

## 4. 기대되는 구조

trace가 감정 표현이라면 다음이 성립해야 한다.

1. `target`, `control_state`, `social_orientation`, `action_tendency`가 비슷한 record는 trace space에서 가까워야 한다.
2. `anger-at-other`는 둘 다 부정 valence를 갖더라도 `guilt/self-blame`과 분리되어야 한다.
3. high arousal과 other-targeted blame은 low control sadness와 다른 response constraint를 보여야 한다.
4. trace space의 nearest neighbor는 우연보다 더 자주 appraisal/action 속성을 공유해야 한다.
5. stimulus text를 고정하더라도 trace field를 바꾸면 생성 응답의 방향이 달라져야 한다.

## 5. 증거로 인정할 수 있는 것

Representation-level evidence:

- baseline보다 높은 nearest-neighbor label consistency
- `target`, `control_state`, `social_orientation`, `action_tendency`에 대한 cluster purity
- 낮은 intra-group distance와 높은 inter-group distance
- bootstrap sample 전반에서 안정적인 cluster

Generation-level evidence:

- trace ablation이 appraisal fidelity를 낮춤
- trace perturbation이 response affect direction을 바꿈
- trace-preserving prompt가 stimulus-only prompt보다 raw affect를 더 잘 유지함

Human-level evidence:

- blind evaluator가 appraisal fidelity와 raw affect preservation에서 trace-conditioned response를 선호함
- evaluator가 trace-conditioned generation에서 intended episode state를 더 정확히 추론함

## 6. 핵심 위험

가장 큰 위험은 현재 trace field가 아직 너무 symbolic하고 수동으로 압축되어 있다는 점이다. 이 경우 prompting에는 도움이 되지만, 견고한 learned emotional geometry를 형성하는 데 실패할 수 있다.

그렇다고 v4가 무효가 되는 것은 아니다. 다만 v3.1에서는 구조화된 label만으로는 부족하고, 더 풍부한 neural trace vector, recurrent trajectory, learned latent state가 필요하다는 뜻이 된다.
