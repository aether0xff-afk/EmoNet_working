# Paper Worklist

## 로컬에서 완료 가능한 작업

- 현재 산출물 기준 데이터 규모, keep rate, consistency 통계 정리
- `z -> s` 회귀기의 간단한 hold-out MAE 및 mean baseline 비교
- 스타일 분포 편향 분석
- 논문 초안, 실험 표, 재현 명령 정리
- 원격 고성능 실험 체크리스트와 실행 명령 준비

## 원격(RDP)에서 돌릴 작업

### 1. End-to-end 성공 로그 재생성
- 목적: 논문에 실패 로그 대신 실제 성공 로그 포함
- 필요 조건: 로컬/원격 OpenAI-compatible LLM 서버 실행
- 권장 명령:

```powershell
python -m emonet.cli e2e-check `
  --text "지금 너무 예민하고 피곤해." `
  --zs-model-path .\artifacts\z_to_s_decoder.npz `
  --base-url "http://127.0.0.1:11434/v1" `
  --model-name "gpt-oss:20b" `
  --report-json .\outputs\validation\e2e_check_report_success.json `
  --output-csv .\outputs\validation\e2e_check_runs_success.csv `
  --log-jsonl .\outputs\validation\e2e_check_runs_success.jsonl
```

### 2. 전체 500 subset 라벨링 확장
- 목적: `(z, s)` 데이터 규모 확장
- 이유: 현재 200개는 논문용으론 작음
- 권장 명령:

```powershell
python -m emonet.cli label-local `
  --input-csv .\outputs\llm\llm_subset.csv `
  --output-csv .\outputs\llm\llm_subset_labeled_500_ollama.csv `
  --base-url "http://127.0.0.1:11434/v1" `
  --model-name "gpt-oss:20b" `
  --limit 500 `
  --block-size 8 `
  --style-dim 32 `
  --generation-temperature 0.4 `
  --rating-temperature 0.0 `
  --max-retries 4 `
  --timeout-sec 180 `
  --keep-threshold 0.18 `
  --keep-failures
```

### 3. z->s 회귀 재학습 및 대규모 예측
- 목적: 더 큰 labeled set 기반 회귀기 업데이트
- 권장 명령:

```powershell
python -m emonet.cli fit-zs-regressor `
  --input-csv .\outputs\llm\llm_subset_labeled_500_ollama.csv `
  --model-path .\artifacts\z_to_s_decoder_500.npz `
  --val-ratio 0.1
```

```powershell
python -m emonet.cli predict-s `
  --input-csv .\outputs\z\out_z_training.csv `
  --output-csv .\outputs\z\out_z_training_with_s_pred_500.csv `
  --model-path .\artifacts\z_to_s_decoder_500.npz
```

### 4. Baseline 생성 실험
- 목적: EmoNet 대비 direct prompting 비교
- 필요 산출물:
  - direct LLM 응답
  - stim-only prompt 응답
  - full style prompt 응답
- 최소 비교 항목:
  - 내용 적합성
  - 스타일 일치도
  - 감정 날것 유지 정도

### 5. Ablation 실험
- 목적: 온화함 편향 원인 분리
- 권장 축:
  - `HS+SS` 입력 vs `HS only`
  - `STYLE_SUMMARY` 포함 vs 제거
  - `STYLE_TAGS` 포함 vs 제거
  - current 32축 vs raw affect 축 확장 버전

## 지금 바로 논문에 넣을 수 있는 현재 결론

- `outputs/z/out_z_training.csv`: 51,628 rows, 60 labels
- `outputs/llm/llm_subset.csv`: 500-row balanced subset
- `outputs/llm/llm_subset_labeled_200_ollama.csv`: 200 rows, 190 keep, keep rate 95%
- current style space is strongly biased toward warmth/politeness/cooperativeness/calmness/softness
- current linear `z -> s` decoder does not yet beat a mean-value baseline on repeated hold-out splits

## 제출 전 꼭 메울 공백

- 성공한 end-to-end 생성 로그
- baseline 대비 정량 결과표
- 최종 응답 품질 평가표
- 관련연구와 참고문헌
