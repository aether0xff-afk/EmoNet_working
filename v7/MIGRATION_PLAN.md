# v7 Migration Plan

## Principle

v6 코드를 통째로 복사한 뒤 조금씩 지우지 않습니다. 그러면 숨은 정책이 남습니다. v7은 최소 runtime을 새로 만들고, v6에서 검증된 관측 장치와 안전 경계만 선택적으로 가져옵니다.

## Phase 0: Audit

Status: active

- [x] v6 인위적 요소 목록 작성
- [x] 제거 대상과 유지할 경계 구분
- [x] 금지 목록 기반 검사기 자체도 meta-rule이므로 제거
- [ ] 학습 목표와 데이터 계약 결정

## Phase 1: Minimal Learned Runtime

- [ ] 입력과 다음 observable output의 데이터 계약 확인
- [ ] 내부 구조를 선행 고정하지 않은 최소 학습 baseline 구성
- [ ] history 저장
- [ ] observer hook 제공
- [ ] 감정 라벨, hormone 이름, persona prompt, 행동 gate 없이 smoke test 작성

## Phase 2: Long-Context Experiment

- [ ] 명시적 memory write gate 제거
- [ ] 명시적 retrieval 정책 제거
- [ ] 장기 의존성 표현 방식 비교
- [ ] 중요도 threshold와 memory type label 제거

## Phase 3: Continuous Output

- [ ] 메시지와 침묵을 동일한 event stream으로 입력
- [ ] `speak`, `wait`를 사람이 정의한 분류로 다루지 않음
- [ ] 다음 observable output을 직접 생성
- [ ] 45분 threshold와 spontaneous reaction label 제거

## Phase 4: Observable Output

- [ ] 내부 구조를 선행 지정하지 않고 observable output 생성 실험
- [ ] style axis와 anti-softening prompt 제거
- [ ] 운영 오류와 접근 제어만 외부 애플리케이션 경계에 유지

## Phase 5: Observer Package

- [ ] 내부 표현이 생기는 경우 geometry probe
- [ ] activation trace exporter
- [ ] interpretability dashboard
- [ ] observer가 runtime 결정을 변경하지 않는지 테스트

## No Meta-Policy

v7은 규칙 제거를 새로운 규칙 목록으로 대체하지 않습니다.

- runtime에 금지 token 검사기를 넣지 않음
- 특정 latent 의미를 금지 목록으로 관리하지 않음
- 사람이 작성한 fallback 응답을 넣지 않음
- observer 결과로 runtime 행동을 교정하지 않음
- 운영 경계를 캐릭터 행동 지침으로 번역하지 않음

## First Experiment Questions

Phase 1 구현 전에 다음 질문을 검토해야 합니다. 답을 문서에서 미리 고정하지 않습니다.

- 어떤 관측 데이터를 사용할 것인가?
- 사람 라벨, 사람 선호, 기존 모델 출력을 각각 어느 범위까지 인위적 개입으로 볼 것인가?
- 다음 observable output을 무엇으로 기록할 것인가?
- 내부 표현이 필요한가, 아니면 관측 가능한 입출력 baseline부터 시작할 것인가?
- 장기 맥락을 데이터에서 어떻게 관측하고 평가할 것인가?
