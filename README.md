# Minecraft RL Agent MVP

네 KSEF 논문의 구조를 Minecraft 환경에 옮긴 최소 실행 버전입니다.

핵심 대응:

- `nmap XML 관측` → `Mineflayer 월드/인벤토리 관측 JSON`
- `KK/KV Knowledge Storage` → 발견 블록, 보유 아이템, 제작 가능성, 실패 원인 저장
- `Policy A/B/C` → WHAT/HOW/WHERE 행동 분해
- `Prophecy Module` → 최근 행동 전이 기반 다음 상태/보상 예측
- `Imagination Cycle` → 실행 전 후보 행동을 롤아웃하여 가장 높은 점수 선택
- `FLAG 발견` → 목표 아이템 제작 또는 획득

## 설치

```bash
npm install
cp config.example.json config.json
npm start
```

## 준비

1. 로컬 Minecraft Java 서버를 켭니다.
2. `server.properties`에서 테스트 편의를 위해 필요하면 `online-mode=false`를 사용합니다.
3. 봇이 접속할 주소와 포트를 `config.json`에 맞춥니다.
4. 실행합니다.

## 현재 목표

기본 목표는 `wooden_pickaxe`입니다.

봇은 다음 행동들을 조합해서 시도합니다.

- WHAT: `observe`, `explore`, `mine`, `craft`
- HOW: `nearest`, `safe`, `random`, `known`
- WHERE: `tree`, `stone`, `self`, `front`, `known_area`

## 로그

실행 로그는 `logs/run-*.jsonl`에 저장됩니다.
각 줄은 한 step의 행동, 상태 변화, 보상, 예언 결과, 상상 사이클 후보 평가를 담습니다.

## 주의

이건 연구용 MVP라서 완전한 Voyager 같은 에이전트가 아닙니다.
처음 목표는 “논문 구조가 Minecraft 환경에서 폐루프로 돌아가는지” 확인하는 것입니다.
