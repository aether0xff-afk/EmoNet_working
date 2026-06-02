# EmoNet

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-research%20prototype-F59E0B)](#project-status)
[![License](https://img.shields.io/badge/license-MIT-22C55E)](LICENSE)

> 감정을 하나의 라벨이 아니라 시간에 따라 변화하는 내부 상태 trace로 다루는 연구 및 프로토타입 저장소

EmoNet은 텍스트 입력을 감정 상태의 변화로 변환하고, 그 변화가 기억, 말투, 행동 경향, 캐릭터 응답에 어떤 영향을 주는지 실험하는 작업 공간입니다. 초기 파이프라인부터 신경 trace 연구, 캐릭터 채팅, Ruca/Rookie 자율 캐릭터 런타임까지의 구현을 버전별로 보존합니다.

현재 안정적인 프로토타입 시작점은 [`v6`](v6)입니다. 사람이 직접 주입한 감정 정책을 제거하는 다음 설계는 [`v7`](v7)에서 진행합니다. 논문 및 평가 파이프라인을 확인하려면 [`v4`](v4), trace 자체가 감정 표현 공간인지 검증하는 실험을 확인하려면 [`v3.1`](v3.1)을 보면 됩니다.

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

`python`이 Windows Store 별칭으로 연결되는 환경에서는 실제 Python 실행 파일 경로를 직접 사용하세요.

## What Runs Today

### Ruca/Rookie autonomous runtime

`v6/ruca_engine`은 사용자 메시지뿐 아니라 침묵도 하나의 이벤트로 처리하는 자율 캐릭터 MVP입니다.

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

짧은 침묵 동안 내부 상태만 갱신:

```powershell
python -m ruca_engine.cli --silence --elapsed-minutes 10 --debug
```

긴 침묵 뒤 조용한 확인 메시지 생성:

```powershell
python -m ruca_engine.cli --elapsed-minutes 60 --llm --debug
```

세션과 기억 유지:

```powershell
python -m ruca_engine.cli "이 대화를 기억해줘" `
  --llm `
  --memory .\outputs\gui\ruca_memory.json `
  --session .\outputs\gui\ruca_session.json `
  --debug
```

`--emonet`을 추가하면 규칙 기반 감정 갱신 대신 v6 artifact를 사용하는 EmoNet trace adapter가 활성화됩니다.

```powershell
python -m ruca_engine.cli "지금 너무 복잡해" --llm --emonet --debug
```

### Local LLM

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
```

API 키, 로컬 세션, 새 대량 산출물은 커밋하지 않습니다.

## Browser GUIs

`v6`에는 목적이 다른 두 개의 로컬 웹 GUI가 있습니다.

| Command | URL | Purpose |
| --- | --- | --- |
| `python .\local_gui.py` | `http://127.0.0.1:8788/` | v5 계열 character-chat service를 확장한 대화 및 AI dialogue 테스트 UI |
| `python .\ruca_gui.py` | `http://127.0.0.1:8790/` | v6 artifact와 영속 세션 파일을 사용하는 간결한 Ruca GUI |

두 GUI는 현재 `emonet.chat_service.generate_chat_turn` 경로를 사용합니다. 침묵 tick과 자율 반응 게이트를 포함한 `ruca_engine.pipeline`을 직접 검증하려면 `python -m ruca_engine.cli`를 사용하세요.

## Architecture

EmoNet 계열의 핵심 연구 흐름:

```text
text
  -> ridge stimulus encoder
  -> 4D stimulus vector
  -> neuron dynamics and branch trace
  -> dominant branch representation z
  -> style projection s
  -> trace-conditioned prompt
  -> LLM response
```

`v6` 자율 캐릭터 경로는 이 trace를 선택적으로 받아 더 높은 수준의 캐릭터 상태로 변환합니다.

```text
EmoNet trace (optional)
  -> valence / arousal / affinity / stability / protective tension / curiosity
  -> memory + relationship context
  -> inner voices
  -> response decision
  -> visible Ruca dialogue
```

주요 모듈:

| Path | Role |
| --- | --- |
| [`v6/ruca_engine/pipeline.py`](v6/ruca_engine/pipeline.py) | 자율 캐릭터 한 턴 전체 orchestration |
| [`v6/ruca_engine/event_scheduler.py`](v6/ruca_engine/event_scheduler.py) | 사용자 메시지, 짧은 침묵, 긴 침묵 이벤트 정규화 |
| [`v6/ruca_engine/emotion.py`](v6/ruca_engine/emotion.py) | 기본 규칙 기반 감정 상태 갱신 |
| [`v6/ruca_engine/emonet_adapter.py`](v6/ruca_engine/emonet_adapter.py) | v5 EmoNet runtime과 v6 artifact를 연결 |
| [`v6/ruca_engine/memory.py`](v6/ruca_engine/memory.py) | 단기, 장기, 관계 기억 JSON 저장 및 검색 |
| [`v6/ruca_engine/inner_voice.py`](v6/ruca_engine/inner_voice.py) | Ruca, Ricky, Rocky 내부 후보 음성 생성 |
| [`v6/ruca_engine/spontaneous.py`](v6/ruca_engine/spontaneous.py) | 발화, 확인 메시지, internal-only 상태 갱신 결정 |
| [`v6/emonet/core.py`](v6/emonet/core.py) | neuron dynamics, branch trace, `z` 표현 계산 |
| [`v6/emonet/chat_service.py`](v6/emonet/chat_service.py) | trace 기반 character-chat 응답 생성 |

## Version Map

이 저장소는 릴리스 패키지 하나가 아니라 연구 계보를 보존하는 working tree입니다.

| Version | Summary | Start Here |
| --- | --- | --- |
| [`v1`](v1) | 텍스트에서 감정 `z`를 만드는 초기 GRU 기반 pipeline과 GUI | [`emotion_z_pipeline.py`](v1/emotion_z_pipeline.py) |
| [`v2`](v2) | encoder, dynamics, clustering, rewiring, branching을 분리한 PyTorch MVP | [`emonet/README.md`](v2/emonet/README.md) |
| [`v3`](v3) | CLI, 실험 스크립트, 평가 산출물을 포함한 self-contained 연구 라인 | [`OUTPUT_LAYOUT.md`](v3/OUTPUT_LAYOUT.md) |
| [`v3.1`](v3.1) | “trace 자체가 감정 상태 표현”이라는 가설을 검증하는 실험 라인 | [`README.md`](v3.1/README.md) |
| [`v4`](v4) | 논문, 평가, trajectory 분석, local GUI가 모인 연구 앱 라인 | [`README.md`](v4/README.md) |
| [`v5`](v5) | EmoNet trace를 캐릭터 발화로 번역하는 character-chat MVP | [`README.md`](v5/README.md) |
| [`v6`](v6) | Ruca/Rookie 자율 캐릭터 런타임과 v6 artifact 통합 | [`README.md`](v6/README.md) |
| [`v7`](v7) | 인위적인 감정 정책을 제거하는 재설계 라인 | [`README.md`](v7/README.md) |

## Repository Tour

```text
.
├── v1/                         # first emotion-z pipeline
├── v2/                         # modular PyTorch MVP
├── v3/                         # legacy research CLI and experiments
├── v3.1/                       # trace-as-emotion representation research
├── v4/                         # paper and evaluation workspace
├── v5/                         # character-chat MVP
├── v6/                         # current Ruca/Rookie runtime
├── v7/                         # de-handcrafted redesign line
├── Dataset/                    # shared Korean emotional-dialogue dataset
├── encoder-LLM-testing/        # LLM label benchmark scripts and results
├── encoder-ML testing/         # classical ML stimulus encoder benchmarks
├── blueprints/                 # design notes and architecture sketches
├── 연구 기록물(수기 등)/       # research notes, reports, and presentation files
├── output/, outputs/           # generated experiment outputs
└── tmp/                        # temporary render and document material
```

### Research and evaluation tools

`v6/scripts`에는 이전 연구 라인에서 이어진 평가 도구가 함께 있습니다.

| Area | Representative scripts |
| --- | --- |
| Trace inspection | `inspect_emotion_trace.py`, `analyze_emotion_trajectory_batch.py`, `analyze_branch_traces.py` |
| Parameter tuning | `branch_param_sweep.py`, `optimize_branch_dynamics.py`, `calibrate_reference_config.py` |
| Response comparison | `experiment_matrix.py`, `score_experiment_matrix.py`, `analyze_paired_superiority.py` |
| Human evaluation | `prepare_human_eval.py`, `analyze_human_eval_results.py` |
| Character dialogue | `character_dialogue_eval.py` |
| Paper artifacts | `paper_metrics.py`, `paper_offline_tables.py`, `generate_paper_svgs.py` |

스크립트별 상세 설명은 [`v6/scripts/README.md`](v6/scripts/README.md)를 참고하세요.

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

이 저장소의 공통 테스트와 연구 스크립트는 `numpy`, `pandas`, `scikit-learn`, `joblib`, `matplotlib`, `torch`를 사용합니다. 먼저 루트 [`requirements.txt`](requirements.txt)를 설치하세요.

## Project Status

이 저장소는 연구 프로토타입입니다.

- 현재 주 개발 프로토타입 경로는 `v6/ruca_engine`입니다.
- `v4`, `v5`, `v6`에는 연구 과정에서 이어진 코드와 스크립트가 일부 중복되어 있습니다.
- `v6/ruca_engine`은 LLM 호출 실패 시 규칙 기반 답변을 임의로 만들어 내지 않고 명시적으로 실패합니다.
- 일부 과거 한국어 문서와 문자열에는 인코딩이 깨진 텍스트가 남아 있습니다. 동작 변경과 별도로 정리해야 합니다.
- 기존 tracked artifact는 재현성을 위해 남겨 두었지만, 새 대량 output과 모델 파일은 `.gitignore`로 차단합니다.

## License

MIT License. 자세한 내용은 [`LICENSE`](LICENSE)를 참고하세요.
