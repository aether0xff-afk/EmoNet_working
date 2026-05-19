<<<<<<< Updated upstream
# Ruca/Rookie MVP Engine

`v6`는 Ruca/Rookie 통합 설계 문서를 실제로 움직이는 최소 엔진으로 옮긴 작업선이다.

현재 구현은 LLM 호출 없이 순수 Python 규칙 기반으로 동작한다. 목표는 최종 대화 품질보다 다음 중간 상태가 실제로 생성되고 검증되는 것이다.

- Ruca/Rookie/Ricky/Rocky 캐릭터 프로필
- 감정 trace 갱신
- 단기/장기/관계 기억 저장
- Ruca/Ricky/Rocky 내부 목소리 후보 생성
- 자발 반응 판단 및 이유 기록
- Ruca 최종 응답 조합

## Run

이 저장소 환경에서는 번들 Python을 쓰는 것이 가장 확실하다.

```powershell
& 'C:\Users\remote\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests -v
```

CLI 스모크 실행:

```powershell
& 'C:\Users\remote\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m ruca_engine.cli "실제로 구현하려면 어떻게 해야 할지 알려줘" --debug
```

메모리와 세션을 파일로 유지하려면:

```powershell
& 'C:\Users\remote\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m ruca_engine.cli "나 지금 너무 불안하고 무서워" --memory .\outputs\ruca_memory.json --session .\outputs\ruca_session.json --debug
& 'C:\Users\remote\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m ruca_engine.cli "이제 조금 정리해줘" --memory .\outputs\ruca_memory.json --session .\outputs\ruca_session.json --debug
```

LLM 응답기로 넘길 프롬프트까지 확인하려면:

```powershell
& 'C:\Users\remote\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m ruca_engine.cli "실제로 구현하려면 어떻게 해야 할지 알려줘" --prompt
```

## Package Layout

- `ruca_engine/models.py`: 핵심 데이터 클래스
- `ruca_engine/profiles.py`: 캐릭터 프로필 로더
- `ruca_engine/emotion.py`: 입력 신호 분석과 감정 trace 갱신
- `ruca_engine/memory.py`: 메모리 저장/조회
- `ruca_engine/inner_voice.py`: 내부 목소리 후보 생성
- `ruca_engine/spontaneous.py`: 자발 반응 판단
- `ruca_engine/composer.py`: Ruca 최종 응답 조합
- `ruca_engine/session.py`: 감정 상태와 최근 턴 히스토리 지속 저장
- `ruca_engine/prompt_builder.py`: LLM 연결용 응답 프롬프트 생성
- `ruca_engine/pipeline.py`: 전체 턴 파이프라인
- `ruca_engine/cli.py`: 단일 턴 실행 CLI
- `data/characters/ruca_rookie_profiles.json`: 기본 캐릭터 프로필

## MVP Boundary

현재 버전은 `Ruca`만 외부 발화자로 사용한다. `Ricky`와 `Rocky`는 내부 목소리 후보로 작동하고, `Rookie`는 사용자/초입자 관점 모델로 유지한다.

향후 LLM 응답 생성이나 v5 trace 런타임을 붙일 때는 `ResponseComposer`와 `EmotionState` 갱신부를 교체하거나 확장하면 된다.

## LLM Composer

LLM final composer를 실제로 켜려면 API 키를 환경변수에 넣고 `--llm`을 붙인다. 기본값은 OpenAI-compatible endpoint다.

```powershell
$env:OPENAI_API_KEY='...'
& 'C:\Users\remote\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m ruca_engine.cli "Ruca처럼 짧게 답해줘" --llm --debug
```

로컬 Ollama/OpenAI-compatible 서버를 쓰려면 `--base-url`과 `--model-name`을 바꾼다. 실패하면 기본적으로 규칙 기반 composer로 fallback되고, 디버그의 `composer_mode`와 `llm_error`에 이유가 남는다.

```powershell
& 'C:\Users\remote\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m ruca_engine.cli "Ruca처럼 짧게 답해줘" --llm --base-url http://localhost:11434/v1 --model-name gpt-oss:20b --debug
```
=======
# EmoNet v6

`v6` is the Ruca & Rookie autonomous character runtime line. It starts from the
`v5` character-chat MVP, then extends it toward characters that keep an
internal emotional life across time, including no-reply ticks, internal voice,
spontaneous response gates, relationship drift, and a Rookie-ready story layer.

The core theme is:

```text
event or silence
  -> EmoNet character tick
  -> memory and relationship update
  -> internal voice
  -> response gate
  -> character message, silence, or internal-only state change
```

`v6` should keep the `v5` principle that the LLM is only the final expression
layer. EmoNet, memory, relationship state, and scene state decide what the
character is carrying before the LLM turns it into dialogue.

## MVP Scope

The first `v6` milestone is Ruca as a single autonomous character:

- user input events and no-reply events are both treated as character events.
- Ruca's internal state can tick even when the user says nothing.
- internal voice is generated as private intermediate state, not shown directly
  to the user.
- a response gate decides between `send_message`, `stay_silent`, and
  `internal_only`.
- emotional memory stores compressed interpretation and deltas, not full raw
  inner monologue.
- Rookie remains a lightweight `scene_state` provider until the autonomous
  Ruca loop is stable.

## Active Code

- `emonet/chat_service.py`: combines EmoNet trace, character context, session
  state, and the LLM call.
- `emonet/character.py`: character card, session memory, felt-state helpers,
  and response validation.
- `local_gui.py`: local browser GUI for character chat and AI dialogue tests.
- `data/characters/default_luca_like.json`: default character card.

## Planned v6 Modules

- `Event Scheduler`: converts user messages, silence, time gaps, and scene
  events into normalized character events.
- `EmoNet Tick Engine`: updates Ruca's affect state from the event, memories,
  traits, relationship state, and scene state.
- `Internal Voice Generator`: creates short private interpretation material
  from the current trace and relationship context.
- `Response Gate`: decides whether Ruca should speak, stay silent, or only
  update internally.
- `Memory Manager`: stores raw conversation references, semantic summaries,
  emotional memories, relationship effects, and slow trait updates.
- `Rookie Plot Manager`: later manages arcs, scenes, conflicts, and
  multi-character coordination without directly forcing character emotions.

## Default LLM

The local GUI defaults to the local Ollama OpenAI-compatible endpoint:

- provider: `openai_compatible`
- endpoint: `http://127.0.0.1:11434/v1`
- model: `gpt-oss:120b-cloud`
- API key: normally empty

You can override the defaults with environment variables:

```powershell
$env:EMONET_LLM_PROVIDER = "openai_compatible"
$env:EMONET_LLM_BASE_URL = "http://127.0.0.1:11434/v1"
$env:EMONET_LLM_MODEL = "gpt-oss:120b-cloud"
$env:EMONET_LLM_API_KEY = ""
```

Check local Ollama models:

```powershell
ollama list
```

## Run

From the repository root, with the Codex bundled Python:

```powershell
$PY = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
cd .\v6
& $PY -m unittest discover -s tests -v
& $PY .\local_gui.py
```

The GUI opens at:

```text
http://127.0.0.1:8788/
```

## Notes

`v6` does not train a new LLM. The LLM is only the final expression layer. The
EmoNet trace, character card, session state, and relationship memory are built
locally and passed into the generation prompt.

For now, most code is inherited from `v5`. New work should land in small
vertical slices: first event/no-reply state, then response gate, then private
internal voice, then Rookie and multi-character expansion.
>>>>>>> Stashed changes
