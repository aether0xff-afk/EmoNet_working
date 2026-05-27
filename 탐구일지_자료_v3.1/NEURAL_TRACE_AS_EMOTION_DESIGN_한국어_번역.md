# Neural Trace-As-Emotion 설계

## 1. 수정된 정의

v3.1에서 `trace`는 stimulus vector가 EmoNet network를 통과할 때 생성되는 neural activation trajectory를 뜻한다.

trace가 주로 의미하는 것은 다음이 아니다.

```text
episode_label
target
control_state
action_tendency
```

위 항목들은 외부 해석 라벨이다.

실제로 연구해야 할 대상은 다음이다.

```text
stimulus vector
-> network propagation
-> tick-by-tick neuron activations
-> dominant branch / branch tensor / z
-> emotion-state trace
```

## 2. 핵심 주장

EmoNet의 강한 주장은 다음과 같다.

> 감정은 stimulus가 neural network를 통과하면서 형성하는 activation trace이다.

symbolic appraisal field는 그 trace를 검증하기 위한 probe로만 유용하다.

## 3. 증거 계획

Representation evidence:

- 같은 appraisal/action label은 비슷한 neural trace를 가져야 한다.
- 다른 label은 neural trace space에서 분리되어야 한다.
- dominant branch는 재사용 가능한 neuron route를 보여야 한다.
- z embedding은 감정 관련 geometry를 보존해야 한다.

Causal evidence:

- neural trace feature를 perturb하면 emotion interpretation이 이동해야 한다.
- 기여도가 높은 neuron을 ablate하면 label separability가 낮아져야 한다.
- neuron count를 늘렸을 때, 추가 capacity가 안정적인 cluster를 형성하는 경우에만 trace geometry가 개선되어야 한다.

## 4. 현재 구현

`scripts/export_neural_activation_traces.py`는 다음을 export한다.

- `activation`: tick x neuron K matrix
- `branch_tensor`: dominant branch feature tensor
- `z`: encoded trace embedding
- `stim_vec`: 원래의 4D stimulus vector
- `dominant_branch_ids`: network를 통과한 route
- `active_counts`: tick별 active neuron count

출력은 다음 위치에 저장된다.

```text
outputs/neural_trace_probe_v1/
```

## 5. 이전 v3.1 작업과의 관계

이전 normalized field는 버리지 않는다. 다만 evaluation label로 위치가 바뀐다.

```text
neural trace = 검증 대상
target/control/social/action labels = 외부 probe
```

이것이 사용자의 원래 아이디어에 맞는 올바른 계층 구조다.
