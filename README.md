# EmoNet 작업 트리

EmoNet은 감정 상태 모델링을 연구하고 프로토타입을 개발하기 위한 작업 공간입니다.
이 저장소에는 여러 시기의 구현 계열이 나란히 보관되어 있으므로 루트 디렉터리는
주로 색인 역할을 합니다. 코드 실행과 테스트는 작업하려는 버전 디렉터리에서
진행하세요.

## 현재 중점 개발 대상

- `v6`: v5를 기반으로 응답 없는 틱(no-reply tick), 내면의 목소리, 자발적 응답
  게이트, Rookie용 장면 및 이야기 상태를 추가한 Ruca & Rookie 자율 캐릭터
  런타임 계열입니다.
- `src/`: KSEF 논문의 구조를 Minecraft 환경에 옮긴 Minecraft RL Agent MVP입니다.
- `v4`: 현재 연구, 평가, 로컬 GUI 개발에 사용하는 계열입니다.
- `v5`: v4 런타임과 v3.1의 흔적 기반 감정(trace-as-emotion) 아이디어를 기반으로
  만든 캐릭터 채팅 MVP입니다.
- `v3.1`: 신경 흔적 자체가 감정 상태 표현이라는 가설을 검증하는 표현 수준의
  실험 계열입니다.

이전 디렉터리는 작업의 연속성을 위해 유지합니다.

- `v1`: 초기 `emotion_z_pipeline` 및 GUI 실험입니다.
- `v2`: 초기 모듈형 PyTorch MVP입니다.
- `v3`: 독립 실행형으로 구성된 이전 연구 및 CLI 계열입니다.

## 디렉터리 구성

```text
.
  src/                 Minecraft RL Agent MVP
  docs/                Minecraft RL Agent 아키텍처 문서
  v1/                  초기 emotion-z 파이프라인
  v2/                  모듈형 MVP
  v3/                  이전 연구 계열
  v3.1/                흔적 기반 감정 실험
  v4/                  현재 연구 및 평가 앱 계열
  v5/                  캐릭터 채팅 MVP 계열
  v6/                  Ruca/Rookie 자율 캐릭터 런타임 계열
  Dataset/             공용 원본 데이터셋
  blueprints/          설계 노트 및 이전 아키텍처 초안
  encoder-LLM-testing/ LLM 레이블링 벤치마크
  encoder-ML testing/  ML 인코더 벤치마크 자료
  output/, outputs/    생성된 그림 및 실험 결과
  tmp/                 문서 및 포스터 빌드용 임시 자료
```

## Python 환경

이 컴퓨터에서는 `python` 명령이 실제 인터프리터 대신 Windows Store 별칭으로
연결될 수 있습니다. Codex에 포함된 Python을 직접 사용하세요.

```powershell
$PY = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $PY --version
& $PY -m pip install -r requirements.txt
```

루트의 공용 의존성 파일에는 일반적인 런타임에 필요한 패키지가 포함되어 있습니다.

```text
joblib, matplotlib, networkx, numpy, pandas, scikit-learn, streamlit, torch
```

Codex 외부에서 일반적인 로컬 환경을 구성하려면 가상 환경을 만들고 동일한 루트
의존성을 설치하세요.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 빠른 시작

v4 테스트 실행:

```powershell
$PY = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
cd .\v4
& $PY -m unittest discover -s tests -v
```

v5 테스트 실행:

```powershell
$PY = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
cd .\v5
& $PY -m unittest discover -s tests -v
```

v6 테스트 실행:

```powershell
$PY = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
cd .\v6
& $PY -m unittest discover -s tests -v
```

v4 로컬 GUI 실행:

```powershell
$PY = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
cd .\v4
& $PY .\local_gui.py
```

v5 캐릭터 채팅 GUI 실행:

```powershell
$PY = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
cd .\v5
& $PY .\local_gui.py
```

v6 Ruca/Rookie GUI 실행:

```powershell
$PY = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
cd .\v6
& $PY .\local_gui.py
```

## Minecraft RL Agent MVP

루트의 Node.js 프로젝트는 KSEF 논문의 구조를 Minecraft 환경에 옮긴 최소 실행
버전입니다.

핵심 대응:

- `nmap XML 관측` -> `Mineflayer 월드/인벤토리 관측 JSON`
- `KK/KV Knowledge Storage` -> 발견 블록, 보유 아이템, 제작 가능성, 실패 원인 저장
- `Policy A/B/C` -> WHAT/HOW/WHERE 행동 분해
- `Prophecy Module` -> 최근 행동 전이 기반 다음 상태/보상 예측
- `Imagination Cycle` -> 실행 전 후보 행동을 롤아웃하여 가장 높은 점수 선택
- `FLAG 발견` -> 목표 아이템 제작 또는 획득

설치 및 실행:

```bash
npm install
cp config.example.json config.json
npm start
```

준비:

1. 로컬 Minecraft Java 서버를 켭니다.
2. `server.properties`에서 테스트 편의를 위해 필요하면 `online-mode=false`를 사용합니다.
3. 봇이 접속할 주소와 포트를 `config.json`에 맞춥니다.
4. 실행합니다.

기본 목표는 `wooden_pickaxe`입니다.

봇은 다음 행동들을 조합해서 시도합니다.

- WHAT: `observe`, `explore`, `mine`, `craft`
- HOW: `nearest`, `safe`, `random`, `known`
- WHERE: `tree`, `stone`, `self`, `front`, `known_area`

실행 로그는 `logs/run-*.jsonl`에 저장됩니다. 각 줄은 한 step의 행동, 상태 변화,
보상, 예언 결과, 상상 사이클 후보 평가를 담습니다.

이건 연구용 MVP라서 완전한 Voyager 같은 에이전트가 아닙니다. 처음 목표는 논문
구조가 Minecraft 환경에서 폐루프로 돌아가는지 확인하는 것입니다.

## API 키

`v5`와 `v6`는 기본적으로 로컬 Ollama OpenAI 호환 엔드포인트를 사용하므로 일반적으로
API 키가 필요하지 않습니다. `v4`에는 Claude용 GUI 경로가 남아 있습니다. Claude 기반
흐름을 실행할 때만 `ANTHROPIC_API_KEY`를 설정하세요.

API 키, 로컬 진행 상태 파일, 생성된 임시 출력물은 커밋하지 마세요.

## Git 관리 원칙

이 저장소에는 선별된 연구 결과물이 의도적으로 포함되어 있습니다. 다만 새로 생성한
대량의 출력물은 신중하게 추가해야 합니다. `.gitignore`는 이후 작업에서 흔히 생성되는
디렉터리와 대용량 모델 및 압축 파일이 추가되지 않도록 차단합니다. 이미 추적 중인
결과물은 별도의 정리 작업에서 의도적으로 이동하거나 삭제할 때까지 계속 추적됩니다.
