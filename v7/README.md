# EmoNet v7

> v6를 계승하되, 사람이 감정의 의미와 행동을 직접 결정하는 규칙을 제거하는 재설계 라인

v7의 목표는 “규칙을 더 정교하게 만드는 것”이 아닙니다. 감정 상태, 기억의 중요도, 발화 시점, 말투가 데이터와 상태 변화에서 나타나도록 만드는 것입니다.

## Current Phase

v7은 지금 **Phase 0: 인위적 요소 감사** 단계입니다.

- 전체 목록: [`ARTIFICIAL_ELEMENTS.md`](ARTIFICIAL_ELEMENTS.md)
- 제거 순서: [`MIGRATION_PLAN.md`](MIGRATION_PLAN.md)

아직 v6 runtime을 복사하지 않았습니다. 먼저 제거 기준을 고정하지 않으면 v6의 규칙을 이름만 바꿔 다시 가져오게 됩니다.

## Definition

v7에서 “인위적 요소”는 사람이 직접 주입한 **모든 행동 결정 규칙**입니다.

제거 대상:

- 특정 단어를 불안, 친밀감, 행동 압력으로 해석하는 키워드 규칙
- 사람이 정한 감정 축과 호르몬 축의 의미
- 사람이 정한 가중치로 상태를 혼합하는 수식
- 특정 임계값에서 기억, 발화, 침묵, 관계 상태를 결정하는 분기
- 캐릭터의 반응을 미리 써 둔 문장과 고정 persona
- 말투를 32개 또는 40개 사람이 정의한 축으로 제한하는 방식
- LLM에게 내부 상태의 해석 방향을 강하게 주입하는 프롬프트
- 특정 반응을 금지하거나 권장하는 token 목록
- 사람이 정한 fallback, retry, 보정, 예외 행동
- “이 상황에서는 이렇게 행동해야 한다”는 안전 연출 규칙
- 새 runtime에 무엇이 들어가면 안 되는지 고정 목록으로 검사하는 방식

runtime 밖에만 둘 수 있는 시스템 경계:

- 프로세스가 죽지 않도록 하는 파일 포맷과 tensor shape 검사
- API timeout, 저장 실패, artifact 불일치 같은 운영 오류 처리
- NaN과 무한대가 시스템을 망가뜨리지 않게 하는 numerical guard
- 접근 권한, 비밀키, 개인정보 저장 범위 같은 애플리케이션 보안 경계
- 실험 재현을 위한 seed, 로그, artifact version

## Open Direction

```text
observation stream
  -> learned model
  -> next observable output
```

내부 구조는 아직 결정하지 않습니다. latent, recurrence, attention, graph는 검토할 수 있는 후보이지 v7의 선행 조건이 아닙니다.

내부 표현이 생기더라도 `alarm`, `warmth`, `dopamine`, `serotonin`, `Ricky`, `Rocky`처럼 사람이 미리 의미를 붙이지 않습니다. 기억 저장, 기억 검색, 발화, 대기처럼 사람이 미리 분리한 행동 gate도 두지 않습니다. 해석은 runtime 정책이 아니라 별도 probe와 분석 도구가 담당합니다. probe의 해석도 runtime으로 역류하지 않습니다.

## Boundary

v7 runtime에는 사람이 작성한 행동 정책이나 행동 분류표를 두지 않습니다. 외부 애플리케이션은 운영 및 보안 경계를 가질 수 있지만, 그것을 감정 상태나 캐릭터 행동으로 번역하지 않습니다.

현재 scaffold는 문서와 빈 [`runtime`](runtime) 디렉터리만 포함합니다.
