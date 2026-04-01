# `(z, s)` Dataset Schema

권장 학습 테이블 컬럼:

- `sample_id`
- `talk_id`
- `label`
- `persona_id`
- `profile_id`
- `text`
- `z_0 ... z_63`
- `dopamine`
- `serotonin`
- `norepinephrine`
- `melatonin`
- `dominant_branch_len`
- `llm_response`
- `s_0 ... s_31`
- `s_hat_0 ... s_hat_31`
- `consistency_l1`
- `keep_sample`

권장 규칙:

- `s_*` 는 generation prompt가 만든 목표 스타일 벡터
- `s_hat_*` 는 rating prompt가 다시 읽어서 만든 평가 벡터
- `llm_response` 는 필요 시 짧은 표정 변화 단서를 포함할 수 있다.
- `consistency_l1 = mean(abs(s - s_hat))`
- `keep_sample = consistency_l1 <= 0.12` 같은 기준으로 필터
