# Picasso Protocol 1.0 공격 실험 설계

이 폴더는 Picasso Protocol 1.0을 암호 알고리즘 관점에서 평가하기 위한 세 가지 공격 실험 설계를 정리한다.

## 포함된 실험

1. **역전파 기반 입력 복원 공격**
   - 이미지에서 추출한 latent vector를 목표로 삼아 원문 입력 embedding을 역최적화한다.
2. **Chosen-Plaintext Attack**
   - 공격자가 선택한 유사 문장들의 출력 latent vector 차이를 비교하여 확산 효과를 분석한다.
3. **Decoder Inversion Attack**
   - 다수의 평문-이미지 쌍으로 대체 decoder를 학습하여 원문 복원 가능성을 평가한다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `section_7_attack_evaluation.md` | 보고서 7장에 바로 넣을 수 있는 공격 실험 설계 문서 |
| `result_templates/backprop_input_recovery_results.csv` | 역전파 기반 입력 복원 공격 결과 기록 양식 |
| `result_templates/chosen_plaintext_results.csv` | Chosen-Plaintext Attack 결과 기록 양식 |
| `result_templates/decoder_inversion_results.csv` | Decoder Inversion Attack 결과 기록 양식 |

## 권장 실험 범위

- 역전파 기반 입력 복원 공격: 500~2,000 optimization steps
- Chosen-Plaintext Attack: 기준 문장 5~10개와 각 기준 문장별 변형 문장
- Decoder Inversion Attack: 최소 100개, 가능하면 1,000개 이상의 평문-이미지 쌍

실험 결과가 낮게 나오거나 공격이 실패하더라도, loss 변화, token accuracy, 출력 예시를 함께 제시하면 안전성 평가 결과로 사용할 수 있다.

## 실제 실행 결과

실험은 `scripts/run_attack_experiments.py`로 재현할 수 있으며, 실행 결과는 `results/`에 저장된다. 현재 저장된 결과의 핵심 요약은 다음과 같다.

| 공격 실험 | 핵심 결과 | 해석 |
|---|---|---|
| 역전파 기반 입력 복원 공격 | 평균 Token Accuracy 1.0000 | latent vector 직접 저장 시 원문 복원 위험이 큼 |
| Chosen-Plaintext Attack | 평균 L2 Distance 2.117602 | 작은 입력 변화가 latent와 PNG 바이트 차이로 관측됨 |
| Decoder Inversion Attack | 최고 Token Accuracy 1.0000 | 평문-latent 쌍이 충분하면 대체 decoder가 원문을 복원 가능 |

주의: 현재 실행 환경에는 PyTorch, Transformers, NumPy, Pillow가 설치되어 있지 않고 네트워크 패키지 설치도 차단되어 있어, 전체 BERT 모델 대신 Python 표준 라이브러리 기반 deterministic toy encoder로 실험을 수행했다. 이 toy encoder는 Picasso Protocol의 핵심 공격면인 `text → tokenizer → per-position latent tensor → PNG payload` 구조를 재현한다.
