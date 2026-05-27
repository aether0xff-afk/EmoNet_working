# v3.1 실험 로드맵

## Phase 1: Static Trace Space Probe

목표: 기존 trace field가 이미 emotion-like structure를 형성하는지 확인한다.

작업:

- numeric/categorical trace vector를 만든다.
- 각 emotional axis에 대해 nearest-neighbor consistency를 계산한다.
- intra-group distance와 inter-group distance를 비교한다.
- 약한 축과 강한 축을 보고한다.

성공 신호:

- `target`, `social_orientation`, `action_tendency` 중 적어도 일부에서 nearest-neighbor consistency가 majority-label baseline보다 명확히 높다.

## Phase 2: Cluster Discovery

목표: emotion label을 직접 사용하지 않아도 cluster가 나타나는지 확인한다.

작업:

- trace vector에 k-means 또는 agglomerative clustering을 수행한다.
- cluster를 `target`, `control_state`, `social_orientation`, `action_tendency`와 비교한다.
- cluster exemplar를 검토한다.

성공 신호:

- cluster가 표면 topic만이 아니라 appraisal/action pattern에 대응한다.

## Phase 3: Trace Ablation

목표: trace가 단순 상관관계가 아니라 causal usefulness를 가진다는 점을 증명한다.

작업:

- full trace로 response를 생성한다.
- 한 번에 하나의 trace axis를 제거한다.
- appraisal fidelity, raw affect preservation, anti-softening, action tendency fit을 평가한다.

성공 신호:

- `target`, `avoid`, `action_tendency`를 제거했을 때 대응 judge metric이 예측 가능한 방식으로 떨어진다.

## Phase 4: Trace Perturbation

목표: trace를 바꾸면 emotional interpretation도 바뀌는지 확인한다.

작업:

- 같은 stimulus text를 유지한다.
- `target`, `control_state`, `avoid`, `action_tendency`를 바꾼다.
- perturbed trace로 response를 생성한다.
- judge 또는 사람 평가자에게 의도한 emotion direction이 바뀌었는지 묻는다.

성공 신호:

- perturbation이 naturalness를 무너뜨리지 않으면서 통제된 변화를 만든다.

## Phase 5: Neural Trace Expansion

목표: neuron 수 증가 또는 learned trace dynamics가 emotion-state geometry를 개선하는지 확인한다.

작업:

- 256, 512, 1024 같은 neuron count를 비교한다.
- branch collapse, trace separability, response fidelity를 평가한다.
- learned cluster가 더 안정적으로 형성되는지 측정한다.

성공 신호:

- larger model이 branch collapse를 다시 일으키지 않으면서 trace separability와 targeted fidelity를 개선한다.

## 보고 규칙

v3.1은 representation evidence와 generation evidence를 분리해 보고해야 한다.

```text
representation proof: trace가 emotion space처럼 작동한다
generation proof: trace가 response를 개선한다
```

이렇게 해야 주장이 너무 이른 단계에서 과도하게 넓어지는 것을 막을 수 있다.
