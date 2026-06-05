# EmoNet

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-research%20prototype-F59E0B)](#project-status)
[![License](https://img.shields.io/badge/license-MIT-22C55E)](LICENSE)

EmoNet은 감정을 하나의 라벨이 아니라 시간에 따라 변하는 내부 상태 trace로 다루는 연구 및 프로토타입 저장소입니다. 초기 감정 파이프라인, trace-as-emotion 실험, character-chat, Ruca/Rookie 자율 캐릭터 런타임, Minecraft RL Agent MVP가 함께 보관되어 있습니다.

현재 주요 시작점은 `v6`입니다. 사람이 직접 주입한 감정 정책을 줄이는 다음 설계는 `v7`에서 진행합니다.

## Quick Start

```powershell
git clone <repository-url>
cd EmoNet_working

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

cd .\v6
..\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Windows Store Python 별칭이 잡힌 환경에서는 실제 Python 실행 파일 경로를 직접 사용하세요.

```powershell
$PY = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $PY --version
```

## Current Focus

- `v6`: v5를 기반으로 no-reply tick, inner voice, spontaneous response gate, Rookie용 장면 및 이야기 상태를 추가한 Ruca/Rookie 자율 캐릭터 런타임입니다.
- `src/`: KSEF 논문의 구조를 Minecraft 환경에 옮긴 Minecraft RL Agent MVP입니다.
- `v5`: EmoNet trace를 캐릭터 발화로 번역하는 character-chat MVP입니다.
- `v4`: 논문, 평가, trajectory 분석, local GUI가 모인 연구 앱 라인입니다.
- `v3.1`: trace 자체가 감정 상태 표현인지 검증하는 실험 라인입니다.

## Ruca/Rookie Runtime

`v6/ruca_engine`은 사용자 메시지뿐 아니라 침묵도 하나의 이벤트로 처리합니다.

```text
user message or silence
  -> event scheduler
  -> emotion state update
  -> memory retrieval
  -> context analysis
  -> Ruca / Ricky / Rocky inner voices
  -> spontaneous response gate
  -> LLM expression layer or internal-only update
  -> session and memory persistence
```

일반 대화 한 턴:

```powershell
cd .\v6
python -m ruca_engine.cli "오늘은 조금 불안해" --llm --debug
```

침묵 동안 내부 상태만 갱신:

```powershell
python -m ruca_engine.cli --silence --elapsed-minutes 10 --debug
```

긴 침묵 뒤 발화 게이트 확인:

```powershell
python -m ruca_engine.cli --elapsed-minutes 60 --llm --debug
```

`--emonet`을 추가하면 규칙 기반 감정 갱신 대신 v6 artifact를 사용하는 EmoNet trace adapter가 활성화됩니다.

```powershell
python -m ruca_engine.cli "지금 너무 복잡해" --llm --emonet --debug
```

## Browser GUIs

`v6`에는 목적이 다른 두 개의 로컬 웹 GUI가 있습니다.

| Command | URL | Purpose |
| --- | --- | --- |
| `python .\local_gui.py` | `http://127.0.0.1:8788/` | v5 계열 character-chat service를 확장한 대화 및 AI dialogue 테스트 UI |
| `python .\ruca_gui.py` | `http://127.0.0.1:8790/` | v6 artifact와 영속 세션 파일을 사용하는 Ruca GUI |

침묵 tick과 자율 반응 게이트를 포함한 `ruca_engine.pipeline`을 직접 검증하려면 `python -m ruca_engine.cli`를 사용하세요.

## Local LLM and API Keys

OpenAI-compatible 서버를 사용할 수 있습니다. 로컬 Ollama 예시:

```powershell
python -m ruca_engine.cli "짧게 답해줘" `
  --llm `
  --base-url http://127.0.0.1:11434/v1 `
  --model-name qwen3:14b `
  --debug
```

외부 API를 사용할 때만 환경 변수에 키를 설정합니다.

```powershell
$env:OPENAI_API_KEY = "..."
$env:ANTHROPIC_API_KEY = "..."
$env:GEMINI_API_KEY = "..."
```

API 키, 로컬 세션, 새 대량 산출물은 커밋하지 않습니다.

## Version Map

| Version | Summary | Start Here |
| --- | --- | --- |
| `v1` | 초기 emotion-z pipeline과 GUI | `v1/emotion_z_pipeline.py` |
| `v2` | encoder, dynamics, clustering, rewiring, branching을 분리한 PyTorch MVP | `v2/emonet/README.md` |
| `v3` | CLI, 실험 스크립트, 평가 산출물을 포함한 self-contained 연구 라인 | `v3/OUTPUT_LAYOUT.md` |
| `v3.1` | trace 자체가 감정 상태 표현이라는 가설을 검증하는 실험 라인 | `v3.1/README.md` |
| `v4` | 논문, 평가, trajectory 분석, local GUI가 모인 연구 앱 라인 | `v4/README.md` |
| `v5` | EmoNet trace를 캐릭터 발화로 번역하는 character-chat MVP | `v5/README.md` |
| `v6` | Ruca/Rookie 자율 캐릭터 런타임과 v6 artifact 통합 | `v6/README.md` |
| `v7` | 인위적인 감정 정책을 줄이는 재설계 라인 | `v7/README.md` |

## Repository Tour

```text
.
  src/                 Minecraft RL Agent MVP
  docs/                Architecture and repository organization docs
  v1/                  Initial emotion-z pipeline
  v2/                  Modular PyTorch MVP
  v3/                  Legacy research CLI and experiments
  v3.1/                Trace-as-emotion representation research
  v4/                  Research, evaluation, and local GUI workspace
  v5/                  Character-chat MVP
  v6/                  Ruca/Rookie autonomous runtime
  v7/                  De-handcrafted redesign line
  Dataset/             Shared Korean emotional-dialogue dataset
  blueprints/          Design notes and architecture sketches
  encoder-LLM-testing/ LLM label benchmark scripts
  encoder-ML testing/  Classical ML stimulus encoder benchmarks
  output/, outputs/    Generated experiment outputs
  tmp/                 Temporary render and document material
```

## Minecraft RL Agent MVP

루트의 Node.js 프로젝트는 KSEF 논문의 구조를 Minecraft 환경에 옮긴 최소 실행 버전입니다.

| Paper concept | Minecraft MVP mapping |
| --- | --- |
| nmap XML observation | Mineflayer world/inventory observation JSON |
| KK/KV Knowledge Storage | 발견 블록, 보유 아이템, 제작 가능성, 실패 원인 저장 |
| Policy A/B/C | WHAT/HOW/WHERE 행동 분해 |
| Prophecy Module | 최근 행동 전이 기반 다음 상태/보상 예측 |
| Imagination Cycle | 실행 전 후보 행동 rollout과 점수화 |
| FLAG discovery | 목표 아이템 제작 또는 획득 |

실행:

```bash
npm install
cp config.example.json config.json
npm start
```

로컬 Minecraft Java 서버를 켠 뒤 `config.json`에 접속 주소와 포트를 맞춥니다. 기본 목표는 `wooden_pickaxe`이며, 로그는 `logs/run-*.jsonl`에 저장됩니다.

## Tests

현재 자율 캐릭터 런타임만 빠르게 확인:

```powershell
cd .\v6
python -m unittest tests.test_ruca_engine -v
```

v6 전체 확인:

```powershell
python -m unittest discover -s tests -v
```

## Git Management

이 저장소에는 선별된 연구 결과물이 의도적으로 포함되어 있습니다. 새로 생성한 대량 출력물, 로컬 세션, 모델 artifact, 압축 파일, API 키는 커밋하지 마세요. 이미 추적 중인 결과물은 별도 정리 작업에서 의도적으로 이동하거나 삭제할 때까지 계속 추적됩니다.

## Project Status

이 저장소는 연구 프로토타입입니다.

- 현재 주 개발 프로토타입 경로는 `v6/ruca_engine`입니다.
- `v4`, `v5`, `v6`에는 연구 과정에서 이어진 코드와 스크립트가 일부 중복되어 있습니다.
- `v6/ruca_engine`은 LLM 호출 실패 시 규칙 기반 답변을 임의로 만들어 내지 않고 명시적으로 실패합니다.
- 일부 과거 한국어 문서와 문자열에는 인코딩이 깨진 텍스트가 남아 있습니다. 동작 변경과 별도로 정리해야 합니다.

## License

MIT License. 자세한 내용은 `LICENSE`를 참고하세요.
