# EmoNet Latest

이 폴더는 대화에서 합의한 설계를 바탕으로 만든 **실행 가능한 PyTorch MVP 코드**입니다.

## 포함된 핵심 요소
- 텍스트 -> 4차원 제어 벡터 `h_t`
- trait EMA
- 구조 기반 클러스터링
- 클러스터 단위 rewiring
- 뉴런 path 기반 branch 추적 / prune / merge / dominant branch 선택
- 전역 history + dominant path 인코딩
- `z -> s` 스타일 회귀
- 규칙 기반 prompt generator
- frozen text regressor 기반 style scorer

## 빠른 실행
```bash
python -m emonet.infer --text "왜 이렇게 일이 많지" --latent_dim 64
```

## 파일 개요
- `config.py`: 전체 설정
- `encoders.py`: 텍스트 인코더, control encoder
- `traits.py`: trait / memory state
- `dynamics.py`: 감정 동역학 코어
- `clustering.py`: 구조 기반 클러스터 관리
- `rewiring.py`: 클러스터 단위 rewiring
- `branching.py`: path branch 추적
- `history_encoder.py`: 전역 히스토리 + dominant path 인코딩
- `tone_regressor.py`: `z -> s`
- `prompt_generator.py`: style -> prompt 제약 생성
- `style_scorer.py`: 32축 style 회귀기
- `model.py`: 전체 통합 wrapper
- `trainer.py`: 최소 학습 스켈레톤
- `infer.py`: CLI 추론

## 주의
- 이 버전은 **MVP / 연구용 골격 코드**입니다.
- pseudo-labeling용 LLM judge 연동은 오프라인 파이프라인으로 분리하는 것이 안전해서, 코드에는 stub/helper 형태만 넣었습니다.
- 구조 기반 community detection은 `networkx.greedy_modularity_communities`를 사용한 근사 구현입니다. Leiden 전용 구현이 필요하면 별도 의존성을 추가하면 됩니다.
