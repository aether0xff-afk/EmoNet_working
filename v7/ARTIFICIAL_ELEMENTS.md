# v6 Artificial Element Audit

이 문서는 v7으로 옮기기 전에 제거하거나 격리해야 할 v6의 인위적 요소를 정리합니다. 인위성은 감정 의미 주입에 한정되지 않습니다. 사람이 runtime의 행동을 규정하는 모든 행위가 제거 대상입니다.

분류:

- **REMOVE**: v7 runtime에 가져오지 않음
- **OBSERVE**: runtime 밖의 probe 또는 분석 도구로 이동
- **BOUNDARY**: runtime 밖의 운영 및 보안 경계로만 유지

## 1. Ruca Engine

| Decision | Artificial element | Evidence | v7 direction |
| --- | --- | --- | --- |
| REMOVE | 불안, 친밀감, 행동 요청, 질문을 regex 키워드로 판정 | `v6/ruca_engine/emotion.py:8-11` | 계승하지 않음 |
| REMOVE | 문자열 길이와 느낌표 개수로 intensity 계산 | `v6/ruca_engine/emotion.py:34-40` | 계승하지 않음 |
| REMOVE | 키워드 적중 시 `0.65`, `0.55`, `0.50`을 넣는 고정 점수 | `v6/ruca_engine/emotion.py:36-40` | 계승하지 않음 |
| REMOVE | valence, arousal, affinity, stability, protective tension, curiosity 전이 수식 | `v6/ruca_engine/emotion.py:53-68` | 계승하지 않음 |
| REMOVE | 45분 이상 침묵이면 발화하는 규칙 | `v6/ruca_engine/event_scheduler.py:42` | 시간도 관측값으로만 입력 |
| REMOVE | distress, implementation request, question, relationship signal 분기 | `v6/ruca_engine/context.py:35-56` | 계승하지 않음 |
| REMOVE | 상위 기억 3개의 importance 합으로 memory pressure 계산 | `v6/ruca_engine/context.py:29` | 별도 memory pressure 개념을 선행 정의하지 않음 |
| REMOVE | 상황별 `user_position`, `rookie_question`, `unresolved_need` 고정 문장 | `v6/ruca_engine/context.py:30-62` | 계승하지 않음 |
| REMOVE | whitespace token overlap과 importance 가산점으로 기억 검색 | `v6/ruca_engine/memory.py:24-47` | 계승하지 않음 |
| REMOVE | alarm, warmth, action pressure 기반 기억 저장 조건과 기억 종류 | `v6/ruca_engine/memory.py:58-65` | 명시적 저장 gate와 기억 종류를 두지 않음 |
| REMOVE | 기억 summary prefix를 상황별로 고정 | `v6/ruca_engine/memory.py:116-125` | 계승하지 않음 |
| REMOVE | Ruca, Ricky, Rocky 역할별 고정 내부 독백 템플릿 | `v6/ruca_engine/inner_voice.py:21-105` | 계승하지 않음 |
| REMOVE | urgency와 confidence를 수식으로 조합 | `v6/ruca_engine/inner_voice.py:31-51` | 수동 점수 및 별도 policy confidence 제거 |
| REMOVE | alarm, warmth, action pressure, 침묵 시간에 따른 반응 gate | `v6/ruca_engine/spontaneous.py:12-55` | 별도 반응 gate를 두지 않음 |
| REMOVE | `check_in`, `warm_reciprocity`, `initiative`, `internal_only` 수동 reaction label | `v6/ruca_engine/spontaneous.py:15-55` | 별도 reaction action space를 선행 정의하지 않음 |
| REMOVE | LLM에 전달하는 감정 축, 내부 음성, 반응 사유 섹션 | `v6/ruca_engine/prompt_builder.py:39-89` | 계승하지 않음 |
| REMOVE | 1~3문장, 따뜻한 반말, 확인 질문 1개 등 발화 연출 규칙 | `v6/ruca_engine/prompt_builder.py:91-101` | 계승하지 않음 |
| REMOVE | EmoNet trace를 6개 사람이 정의한 EmotionState 축으로 변환 | `v6/ruca_engine/emonet_adapter.py:69-93` | 계승하지 않음 |
| REMOVE | Ruca, Rookie, Ricky, Rocky의 수동 traits와 관계 설명 | `v6/data/characters/ruca_rookie_profiles.json` | 계승하지 않음 |

## 2. EmoNet Core

| Decision | Artificial element | Evidence | v7 direction |
| --- | --- | --- | --- |
| REMOVE | 4D 입력을 dopamine, serotonin, norepinephrine, melatonin으로 명명 | `v6/emonet/core.py:714`, `v6/emonet/legacy_cli.py:377` | 계승하지 않음 |
| REMOVE | 긍정, 성취, 통제, 안정, 안전, 위협, 경보, 피로, 휴식 키워드 사전 | `v6/emonet/core.py:493-621` | 계승하지 않음 |
| REMOVE | 키워드 비율과 수동 계수로 proxy hormone target 생성 | `v6/emonet/core.py:745-775` | 계승하지 않음 |
| REMOVE | 억제성, 흥분성, 조절성 neuron type 비율 | `v6/emonet/core.py:163-170` | 계승하지 않음 |
| REMOVE | neuron type별 intrinsic bias 중심값 | `v6/emonet/core.py:792-799` | 계승하지 않음 |
| REMOVE | 활성 임계값, 기억 임계값, decay, refractory tick | `v6/emonet/core.py:181-184` | 해당 의미를 가진 수동 동역학을 계승하지 않음 |
| REMOVE | memory mix, hysteresis, fatigue, homeostasis 수동 gain | `v6/emonet/core.py:188-218` | 해당 의미를 가진 수동 모듈을 계승하지 않음 |
| REMOVE | dopamine=rewire, serotonin=prune, melatonin=dropout, norepinephrine=threshold shift 의미 연결 | `v6/emonet/core.py:1814-1834` | 이름 붙은 edge update rule을 두지 않음 |
| REMOVE | top-k 입력 선택, top-k branch 선택, branch 길이 bonus | `v6/emonet/core.py:185`, `v6/emonet/core.py:231-233`, `v6/emonet/core.py:1275`, `v6/emonet/core.py:1613-1618` | 계승하지 않음 |
| OBSERVE | branch trace, activation history, convergence 로그 | `v6/emonet/core.py:128-136`, `v6/emonet/core.py:1676-1690` | 과거 연구 기록으로만 보존 |
| BOUNDARY | tensor shape 검증, finite range 방어, artifact load 실패 | `v6/emonet/core.py` | runtime 밖 numerical and interface boundary |

## 3. Character Chat Runtime

| Decision | Artificial element | Evidence | v7 direction |
| --- | --- | --- | --- |
| REMOVE | 고정 persona, speech style, trigger map, boundary rule, relationship stage | `v6/data/characters/default_luca_like.json` | 수동 캐릭터 반응 정의 제거 |
| REMOVE | 사람이 정의한 32개 style axis와 extended axis | `v6/emonet/core.py:49-82`, `v6/emonet/legacy_cli.py:1495-1532` | 계승하지 않음 |
| REMOVE | style, raw trace, appraisal trace, episode trace 등 수동 conditioning mode | `v6/emonet/chat_service.py:38-46` | 계승하지 않음 |
| REMOVE | raw signal을 8개 이름 붙은 축으로 LLM에게 추출시킴 | `v6/emonet/chat_service.py:56-124` | 계승하지 않음 |
| REMOVE | raw signal 8개를 hormone 4개로 변환하는 수동 수식 | `v6/emonet/chat_service.py:350-369` | 계승하지 않음 |
| REMOVE | 물리 행동, 경계 압력, 상호성에 대한 수동 보정 수식 | `v6/emonet/chat_service.py:385-457` | runtime 행동 보정에서 제거 |
| REMOVE | 이전 상태 carryover blend, relation load, decay, soft cap | `v6/emonet/chat_service.py:553-760` | 계승하지 않음 |
| REMOVE | trace를 felt state로 다시 해석하는 수동 규칙 | `v6/emonet/chat_service.py:781-923` | observer probe로 이동 |
| REMOVE | translation surface의 line shape, action texture, pacing 수동 결정 | `v6/emonet/chat_service.py:925-1185` | 계승하지 않음 |
| REMOVE | relationship 상태를 고정 문장으로 덮어쓰기 | `v6/emonet/character.py:192-195` | 계승하지 않음 |
| REMOVE | response 내부 구조 노출 여부를 고정 금지 token으로 검사 | `v6/emonet/character.py:12-37`, `v6/emonet/character.py:376-424` | runtime 행동 규칙으로 사용하지 않음 |
| BOUNDARY | 비밀키, 개인정보, 내부 시스템 데이터의 외부 유출 차단 | 애플리케이션 계층 | 캐릭터 행동과 분리된 접근 제어 및 데이터 경계 |

## 4. Prompt-Level Steering

| Decision | Artificial element | Evidence | v7 direction |
| --- | --- | --- | --- |
| REMOVE | anti-softening, grounding, appraisal 해석 지시 | `v6/emonet/episode_conditioning.py`, `v6/emonet/legacy_cli.py:2443-2986` | 계승하거나 학습 데이터에 우회 주입하지 않음 |
| REMOVE | 최근 대화는 참고만 하고 마지막 입력을 우선하라는 고정 규칙 | `v6/emonet/chat_service.py:282-301` | 계승하지 않음 |
| REMOVE | agent perception JSON schema와 세부 해석 규칙 | `v6/emonet/chat_service.py:56-124` | 계승하지 않음 |
| REMOVE | 응답 retry 때 말투와 형식을 재주입하는 문구 | `v6/emonet/chat_service.py:53-59`, `v6/emonet/llm_api.py:238-300` | 운영 실패만 외부에서 보고하고 행동 보정은 하지 않음 |
| OBSERVE | style generation/rating prompt | `v6/prompts/style_generation_prompt.md`, `v6/prompts/style_rating_prompt.md` | 과거 연구 기록으로만 보존 |
| REMOVE | response generation prompt의 연출 지침 | `v6/prompts/response_generation_prompt.md` | 계승하지 않음 |

## 5. Operational Defaults

다음 값도 사람이 정합니다. 완전히 없앨 수는 없지만 runtime 행동 의미로 사용하지 않고, 실험 실행 환경의 설정으로 격리합니다. 특정 내부 구조를 채택한다는 뜻은 아닙니다.

| Decision | Element | Examples |
| --- | --- | --- |
| CONFIGURE | 모델 용량 | parameter count, context window |
| CONFIGURE | 실행 제한 | timeout, storage quota |
| CONFIGURE | 샘플링 | temperature, seed, validation split |
| CONFIGURE | 저장 위치 | artifact path, dataset path, output path |
| CONFIGURE | numerical safety | clipping, epsilon, finite checks |

## 6. External Boundary

v7 runtime은 안전을 연출하지 않습니다. 사람이 정한 안전 행동도 캐릭터 정책에 넣지 않습니다.

다만 runtime 밖 애플리케이션에는 시스템을 운영하기 위한 경계가 필요합니다. 이것은 캐릭터 행동 규칙이 아니라 실행 환경의 접근 제어입니다.

- API key 보호
- 개인정보 및 세션 저장 범위 제어
- NaN, 무한대, shape mismatch, artifact version mismatch 차단
- LLM 실패 시 조용히 임의 답변을 꾸며 내지 않고 명시적으로 실패
- 허용되지 않은 데이터 접근 차단

출력 내용을 특정 방향으로 교정하는 금지어 목록, 동의 상황별 반응 문구, fallback 답변, 감정 보정 규칙은 runtime에 두지 않습니다.

## 7. Removal Order

1. `ruca_engine`의 regex, 고정 문장, threshold gate 제거
2. character-chat의 persona prompt, style axis, 수동 carryover 제거
3. hormone 이름과 proxy target 제거
4. neuron dynamics의 수동 gain, 수동 모듈, 사람이 정한 의미 연결 제거
5. 해석 도구를 runtime 밖 observer package로 이동
6. 운영 및 보안 경계는 runtime 밖 애플리케이션 계층으로 분리

## 8. Meta-Rules Are Also Artificial

v7에 허용되는 구현 패턴을 사람이 고정 목록으로 정의하고 자동 검사하는 행위도 최종 runtime 설계에는 포함하지 않습니다.

초기 감사 문서는 제거 작업을 위한 임시 지도입니다. 학습된 runtime이 완성되면 이 문서는 역사 기록으로 남고, runtime 행동을 제한하는 실행 정책으로 사용되지 않습니다.
