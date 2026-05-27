# Ruca/Rookie MVP Engine

`v6`는 업로드된 Ruca & Rookie 통합 설계를 실제로 움직이는 최소 엔진으로 옮기는 작업선이다.

핵심 목표는 단순 캐릭터 챗봇이 아니라, 입력과 무입력 시간을 모두 사건으로 처리하고 감정 trace, 기억, 내부 목소리, 응답 gate를 통해 Ruca가 관계적 반응을 만들게 하는 것이다. LLM은 캐릭터 그 자체가 아니라 최종 문장 생성 계층으로 둔다.

## Current MVP

현재 구현은 다음 상태를 실제 객체와 디버그 로그로 만든다.

- Ruca/Rookie/Ricky/Rocky 캐릭터 프로필
- 사용자 입력 이벤트와 `no_reply` 무입력 이벤트
- 감정 trace 갱신
- 단기/장기/관계/감정 기억 저장
- Ruca/Ricky/Rocky 내부 목소리 후보 생성
- 자발 반응 판단과 `send_message`/`stay_silent`/`update_internal_only` 응답 gate
- 캐릭터 trait EMA 업데이트
- Rookie scene/plot pressure와 unresolved thread 추적
- user/Ruca/Ricky/Rocky/Rookie 관계 graph 누적
- Ruca/Ricky/Rocky 표면 화자 선택
- Ruca 최종 응답 조합

기본 표면 화자는 Ruca다. 다만 분석/구조화 요청은 Ricky, 강한 실행/긴급 행동 요청은 Rocky가 제한적으로 표면 화자로 선택될 수 있다. Rookie는 직접 발화자보다 scene/plot 관점을 제공하는 계층으로 유지한다.

## Package Layout

- `ruca_engine/models.py`: 핵심 데이터 클래스
- `ruca_engine/profiles.py`: 캐릭터 프로필 로더
- `ruca_engine/emotion.py`: 입력/무입력 이벤트 신호 분석과 감정 trace 갱신
- `ruca_engine/memory.py`: 메모리 저장/조회 및 Ruca 해석/감정 delta 저장
- `ruca_engine/context.py`: Rookie 관점을 포함한 턴 맥락 분석
- `ruca_engine/inner_voice.py`: 내부 목소리 후보 생성
- `ruca_engine/spontaneous.py`: 자발 반응 판단
- `ruca_engine/response_gate.py`: 표면 메시지/침묵/내부 업데이트 결정
- `ruca_engine/trait_state.py`: 캐릭터 trait EMA 상태 갱신
- `ruca_engine/plot_manager.py`: Rookie plot state와 장면 압력 관리
- `ruca_engine/relationship_graph.py`: 관계 edge와 지표 누적
- `ruca_engine/character_runtime.py`: 표면 화자 선택
- `ruca_engine/composer.py`: Ruca 최종 응답 조합
- `ruca_engine/session.py`: 감정 상태와 최근 턴 히스토리 지속 저장
- `ruca_engine/prompt_builder.py`: LLM 연결용 응답 프롬프트 생성
- `ruca_engine/pipeline.py`: 전체 이벤트 파이프라인
- `ruca_engine/cli.py`: 단일 턴 실행 CLI
- `data/characters/ruca_rookie_profiles.json`: 기본 캐릭터 프로필

## Run

저장소 루트에서:

```powershell
python -m unittest discover -s v6/tests -v
```

CLI 스모크 실행:

```powershell
cd .\v6
python -m ruca_engine.cli "실제로 구현하려면 어떻게 해야 할지 알려줘" --debug
```

메모리와 세션을 파일로 유지하려면:

```powershell
cd .\v6
python -m ruca_engine.cli "나 지금 너무 불안하고 무서워" --memory .\outputs\ruca_memory.json --session .\outputs\ruca_session.json --debug
python -m ruca_engine.cli "이제 조금 정리해줘" --memory .\outputs\ruca_memory.json --session .\outputs\ruca_session.json --debug
```

무입력 시간을 사건으로 tick하려면:

```powershell
cd .\v6
python -m ruca_engine.cli --event-type no_reply --elapsed-minutes 45 --memory .\outputs\ruca_memory.json --session .\outputs\ruca_session.json --debug
python -m ruca_engine.cli --event-type no_reply --elapsed-minutes 180 --memory .\outputs\ruca_memory.json --session .\outputs\ruca_session.json --debug
```

짧은 무입력은 `update_internal_only`로 세션만 갱신하고, 긴 무입력과 내부 압력이 충분한 경우에만 `send_message`가 선택된다.

## LLM Composer

LLM final composer를 실제로 켜려면 API 키를 환경변수에 넣고 `--llm`을 붙인다. 기본값은 OpenAI-compatible endpoint다.

```powershell
$env:OPENAI_API_KEY='...'
cd .\v6
python -m ruca_engine.cli "Ruca처럼 짧게 답해줘" --llm --debug
```

로컬 Ollama/OpenAI-compatible 서버를 쓰려면 `--base-url`과 `--model-name`을 바꾼다.

```powershell
python -m ruca_engine.cli "Ruca처럼 짧게 답해줘" --llm --base-url http://localhost:11434/v1 --model-name gpt-oss:20b --debug
```

`response_gate`가 `update_internal_only` 또는 `stay_silent`를 선택한 경우 LLM composer는 호출하지 않는다. 게이트가 표면 메시지를 보내기로 한 경우에만 LLM이 최종 문장을 만든다.

## Runtime State

세션 파일에는 다음 상태가 함께 저장된다.

- `emotion_state`: Ruca의 현재 감정 trace
- `trait_state`: 캐릭터별 trait EMA
- `plot_state`: Rookie scene pressure와 unresolved threads
- `relationship_graph`: user/Ruca/Rookie/Ricky/Rocky 관계 edge
- `recent_history`: 최근 사용자/표면 응답 이벤트

디버그 JSON에는 위 상태와 함께 `visible_speaker`, `response_decision`, `spontaneous_reaction`, `inner_voices`, `saved_memory`가 기록된다.

## Design Boundary

현재 버전은 PDF의 1단계에 맞춘다.

- 기본 외부 발화자는 Ruca이며, Ricky/Rocky 직접 발화는 분석/실행 압력이 명확할 때만 허용한다.
- 무입력 시간도 사건으로 처리한다.
- 내부 목소리는 사용자에게 그대로 보이지 않는다.
- 메모리는 대화 원문만이 아니라 Ruca의 해석, 감정 delta, 관계 효과를 함께 저장한다.
- Rookie는 장기 플롯/scene 계층으로 확장할 수 있게 맥락 질문을 제공한다.
- 이번 v6 런타임 통합의 사용자 표면은 CLI다. GUI 파일은 기존 흐름과 분리해 두고 이 단계에서 v6 `RucaPipeline`에 연결하지 않는다.
