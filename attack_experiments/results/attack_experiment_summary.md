# Picasso Protocol 1.0 공격 실험 실행 결과

## 실행 조건

- 실행 스크립트: `attack_experiments/scripts/run_attack_experiments.py`
- 외부 패키지 없이 Python 표준 라이브러리만 사용
- 전체 BERT/PyTorch 모델 대신, `text → tokenizer → per-position latent tensor → PNG payload` 구조를 갖는 deterministic toy encoder 사용
- 목적: latent vector를 PNG에 직접 저장하는 구조가 공격자에게 얼마나 많은 정보를 노출하는지 재현 가능한 방식으로 확인

## 핵심 결과

- 역전파 기반 입력 복원 공격 평균 token accuracy: **1.0000**
- Chosen-Plaintext Attack 평균 L2 distance: **2.117602**
- Decoder Inversion Attack 최고 token accuracy: **1.0000** (dataset size 500)

## 해석

1. PNG에서 latent vector를 직접 추출할 수 있으면, 입력 복원 공격이 매우 강하게 동작한다.
2. 유사 문장 사이에서도 latent distance와 PNG byte difference가 측정 가능하므로 chosen-plaintext 분석이 가능하다.
3. 평문-latent 쌍을 모을 수 있으면 대체 decoder가 원문 토큰을 높은 정확도로 복원할 수 있다.
4. 후속 버전에서는 latent vector를 이미지에 평문으로 저장하지 말고, 키 기반 암호화와 per-input salt를 적용해야 한다.
