# EmoNet v5 Clean Trace Validation

EmoNet v5의 목적은 감정을 직접 분류하거나 감정 축을 입력에 주입하는 것이 아니다.

핵심 질문은 다음 하나에서 시작한다.

> 감정 라벨이나 hormone/appraisal 축을 입력하지 않아도, 상태를 유지하는 recurrent dynamics의 trace가 현재 텍스트만으로는 설명되지 않는 과거 맥락 정보를 담는가?

기존 `v5/`는 GPT가 확장한 character-chat 계열의 legacy 구현으로 그대로 보존한다. 이 폴더는 v1~v4 감사 이후 다시 설계한 새 연구선이다.

## 설계 원칙

1. **No affect labels in the core**
   - dopamine / serotonin / norepinephrine / melatonin 없음
   - valence / arousal / emotion class 없음
   - anger/sadness 등의 handcrafted keyword rule 없음
2. **Frozen input representation**
   - 실제 실험은 LM Studio/OpenAI-compatible embedding endpoint 사용 가능
   - 테스트용 deterministic hashing encoder는 의미 성능 주장을 위한 것이 아니라 CI/smoke 전용
3. **Fixed recurrent topology first**
   - baseline에서는 rewiring/plasticity 없음
   - 효과가 확인된 뒤 adaptation, memory, plasticity를 하나씩 ablation으로 추가
4. **Raw trace first**
   - trace를 먼저 직접 저장하고 검증
   - `z`, style, response generation은 core proof 이후 단계
5. **Explicit reset semantics**
   - `reset_transient()`: 수집된 trace만 비우고 recurrent state는 유지
   - `reset_episode()`: recurrent state와 trace를 초기화
   - `reset_all()`: seed 기준으로 topology/projection까지 재생성
6. **Controls are first-class**
   - real trace
   - temporally shuffled trace
   - wrong-sample trace
   - reset-history trace

## 현재 baseline

```text
text event
   ↓
frozen encoder
   ↓
fixed random projection
   ↓
leaky recurrent dynamics
   ↓
tick-by-tick raw trace
```

현재 baseline은 학습 모델이 아니다. 먼저 **state persistence 자체가 올바르게 작동하고, history-dependent trace를 재현 가능하게 생성하는지** 검증하기 위한 최소 substrate다.

## 첫 acceptance gate

같은 마지막 문장 `X`에 대해 서로 다른 과거 맥락 `A`, `B`를 준다.

```text
A → X  => trace_A(X)
B → X  => trace_B(X)
```

다음이 성립해야 한다.

1. history를 유지하면 `trace_A(X) != trace_B(X)`
2. 마지막 문장 전에 `reset_episode()`를 하면 두 trace가 다시 같아진다.
3. 같은 seed와 같은 sequence는 bitwise에 가까운 deterministic trace를 재현한다.
4. shuffled/wrong-trace control이 동일 shape으로 생성된다.

이 gate는 "감정"을 증명하지 않는다. 단지 EmoNet 내부 state가 현재 입력 이외의 과거 정보를 보존할 수 있는지 확인한다.

## 다음 단계

Stage A — dynamics sanity
- collapse / saturation / reproducibility

Stage B — context memory
- same-current-input / different-history benchmark
- history reset ablation

Stage C — usefulness
- text only
- trace only
- text + real trace
- text + shuffled trace
- text + wrong-sample trace
- text + reset trace

Stage D — affect probe
- core training에는 emotion label을 쓰지 않음
- frozen trace에 대해서만 downstream affect probe 수행
- text-only baseline 대비 incremental information 측정

## 빠른 시작

```powershell
cd v5_clean
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
pytest -q
python experiments/run_context_smoke.py
```

LM Studio embedding을 쓰려면 `LMStudioEmbeddingEncoder`에 endpoint와 embedding model을 지정한다.
