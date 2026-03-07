# EmoNet_working

## 상세 설계 계획 (raw_vec 회귀 모델 교체 + 나머지 논문 구조 유지)

이 계획은 **문장 → 감정 벡터(raw_vec)만 회귀 모델로 교체**하고, SNN/IPT/가소성/current_mood/페르소나는 **논문 구조 그대로 유지**하는 것을 목표로 합니다.

### 목표 스펙 고정
- 출력 벡터: `raw_vec = [D, S, NE, M]` (각 0~1 범위)
  - D: 보상/기대
  - S: 안정/신뢰
  - NE: 긴장/스트레스
  - M: 피로/회피
- 추론 인터페이스:
  - 입력: `user_utterance: str` (+ 선택: `dialog_context: list[str]`)
  - 출력: `raw_vec: np.ndarray shape (4,)` in `[0,1]`

### Phase 0) 리포 구조 및 스펙 정리
- 모듈 경계 확정:
  - `raw_vec_model/` : 텍스트 → raw_vec 회귀 모델
  - `snn_engine/` : raw_vec → brain_vec (논문 구조)
  - `ipt_memory/` : 감정 흔적 스택 (논문 구조)
  - `plasticity/` : 구조적 가소성 (논문 구조)
  - `mood_state/` : current_mood 누적 (논문 구조)
  - `persona/` : LLM 페르소나 프롬프트 (논문 구조)
- 인터페이스 고정:
  - `predict_raw_vec(text, context=None) -> np.ndarray(4,)`
  - `snn_step(raw_vec) -> brain_vec`
  - `update_current_mood(brain_vec) -> mood_vec`

### Phase 1) 라벨링 및 데이터 파이프라인
- 라벨 데이터 설계:
  - `text: str`
  - `labels: [D, S, NE, M]`
  - (선택) `context_prev_k: list[str]`
- 라벨링 전략:
  - v1: 수동 라벨링 샘플 구축
  - v2: 약지도(규칙 기반) + 일부 검수
  - v3: LLM 라벨링(훈련용만 사용)

### Phase 2) 회귀 모델 베이스라인 구현
- 모델: TF-IDF + Ridge/Lasso (멀티타깃 회귀)
- 학습:
  - Loss: MSE 또는 Huber
  - 출력: sigmoid 또는 학습 후 0~1 클램프
- 평가:
  - 축별 MAE/MSE
  - 전체 cosine similarity
- API 래퍼:
  - `predict_raw_vec(text)` 고정 인터페이스 제공

### Phase 3) SNN 엔진 연결 (논문 구조 유지)
- raw_vec 입력만 교체하고 SNN 구조는 논문대로 유지
- 핵심 요소:
  - 160 뉴런(흥분 100 / 억제 30 / 조절 30)
  - 5% 연결
  - tick 전파, spike/불응기, 가중 평균 결합

### Phase 4) current_mood 누적 + 페르소나
- brain_vec 기반 장기 감정 누적(current_mood)
- 감정 상태를 페르소나 프롬프트에 반영

### Phase 5) IPT & 구조적 가소성
- IPT 스택 기반 감정 흔적 구현
- ΔW 기반 강화/약화 및 연결 생성/삭제 적용

### Phase 6) 모델 고도화 (선택)
- KoBERT/RoBERTa 계열 + 회귀 헤드로 업그레이드
- 최근 k턴 입력 포함 (문맥형)
- 불확실도 추정 → SNN 억제성 바이어스 강화 등 반영

## 작업 시작: 초기 체크리스트
1. 라벨링 가이드라인 1페이지 문서 작성
2. 데이터 포맷(JSONL/CSV) 템플릿 작성
3. 베이스라인 학습/추론 코드 뼈대 생성
4. SNN 엔진과의 I/O 연결 규격 검증

필요한 경우 위 체크리스트부터 순서대로 구현을 진행합니다.
