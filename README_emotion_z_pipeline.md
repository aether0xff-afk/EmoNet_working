# Emotion Z Pipeline

이 파일은 **텍스트 -> 자극 인코더 -> 감정 동역학 시뮬레이터 -> 히스토리 H -> GRU 기반 히스토리 인코더 -> z** 까지만 구현한 버전이다.

## 포함된 것
- Ridge + TF-IDF 기반 텍스트 점수 예측기
- 8차원 appraisal stimulus `u`
- 4차원 hormone-like control `h`
- 흥분성 / 억제성 / 조절성 뉴런을 가진 감정 동역학 네트워크
- 기억 저장 / 시간 감쇠 / pruning / 항상성 / rewiring / 약한 modulation
- `H -> z` GRU 인코더
- self-supervised 학습:
  - `z -> summary target`
  - `z -> short future target`

## 출력 텐서
- `u in R^8`
- `h in R^4`
- `H in R^(T x d_h)` where `d_h = 16`
- `z in R^(16)` by default

## 학습 예시
```bash
python emotion_z_pipeline.py train \
  --dataset_csv /mnt/data/dataset_for_regression.csv \
  --benchmark_csv /mnt/data/benchmark_results_20260305_180830.csv \
  --label_map_csv /mnt/data/label_map.csv \
  --model_out /mnt/data/emotion_z_pipeline.pt \
  --history_train_samples 512 \
  --epochs 5
```

## 추론 예시
```bash
python emotion_z_pipeline.py infer \
  --model_path /mnt/data/emotion_z_pipeline.pt \
  --text "왜 이렇게 일이 많지 너무 지친다"
```

## 메모
- 현재 버전은 `z`까지만 구현했다.
- `z -> 말투(s)` 변환은 의도적으로 넣지 않았다.
- 동역학 모델은 해석 가능한 규칙 기반 simulator 성격이 강하다.
