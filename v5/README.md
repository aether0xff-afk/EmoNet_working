# EmoNet v5

`v5`는 `v4`의 실행 가능한 EmoNet runtime을 기반으로 시작한 캐릭터 대화 작업선이다.

목표는 자체 LLM을 새로 학습하는 것이 아니라, 기존 LLM 위에 EmoNet 감정 trace, 캐릭터 카드, 세션 기억, 관계 상태를 얹어 Luca형 캐릭터 대화 MVP를 만드는 것이다.

## Active 코드

- `emonet/chat_service.py`: EmoNet trace + 캐릭터 컨텍스트를 합쳐 대화 응답 생성
- `emonet/character.py`: 캐릭터 카드, 세션 상태, v3.1 trace-as-emotion 프롬프트 래퍼
- `local_gui.py`: 캐릭터 대화 중심 로컬 GUI
- `data/characters/default_luca_like.json`: 기본 캐릭터 카드

## Run

```powershell
cd .\v5
python -m unittest discover -s tests -v
python .\local_gui.py
```

기본 GUI는 `http://127.0.0.1:8788/`에서 열린다.

## Claude API

v5의 기본 API provider는 Anthropic Messages API이며, 기본 모델은 Claude Haiku 4.5다.

- 기본 모델: `claude-haiku-4-5-20251001`
- 기본 endpoint: `https://api.anthropic.com/v1/messages`
- API key: GUI 왼쪽 입력칸 또는 `ANTHROPIC_API_KEY`
- 비용 추정 기본값: input `$1/MTok`, output `$5/MTok`

환경변수로 런타임 기본값을 바꿀 수 있다.

```powershell
$env:ANTHROPIC_API_KEY="..."
$env:EMONET_CLAUDE_MODEL="claude-haiku-4-5-20251001"
$env:EMONET_CLAUDE_INPUT_PRICE="1.0"
$env:EMONET_CLAUDE_OUTPUT_PRICE="5.0"
python .\local_gui.py
```

## v3.1 반영점

v3.1의 핵심 관점인 `trace is the emotion-state representation itself`를 v5 캐릭터 프롬프트에 반영했다. 즉 trace는 설명용 메타데이터가 아니라 캐릭터가 현재 느끼는 내부 정서 상태로 취급된다.
