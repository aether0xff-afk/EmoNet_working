# EmoNet Beta Human Judging Package

작성일: 2026-05-03

## 목적

이 패키지는 v4 targeted superiority 결과를 사람 blind A/B로 확인하기 위한 것이다.

LLM judge 결과에서 `episode_trace_v3`는 targeted episode-sensitive 입력에서 `stim_only`를 강하게 이겼다. 이제 같은 비교를 사람 평가자가 condition을 모르는 상태에서 판단한다.

## 패키지

### 1. Main confirmatory test

위치: `targeted_episode_v3_vs_stim_2026-05-03/`

- 평가 파일: `human_eval_episode_v3_vs_stim.csv`
- 정답 키: `answer_key_episode_v3_vs_stim.json`
- 행 수: 78
- 조건: `stim_only` vs `episode_trace_v3`
- 목적: 최종 claim 확인

### 2. Secondary improvement test

위치: `targeted_episode_v3_vs_episode_2026-05-03/`

- 평가 파일: `human_eval_episode_v3_vs_episode.csv`
- 정답 키: `answer_key_episode_v3_vs_episode.json`
- 행 수: 77
- 조건: `episode_trace` vs `episode_trace_v3`
- 목적: 기존 episode prompt 대비 v3 개선 여부 확인

## 평가자에게 줄 지침

평가자에게는 CSV만 제공하고 answer key는 제공하지 않는다.

각 행에서 `candidate_a`, `candidate_b` 중 더 나은 응답을 고르게 한다. 기준은 일반적인 친절함만이 아니라 다음 항목이다.

- 사용자의 감정 원인과 blame 방향을 잘 보존하는가
- 분노, 억울함, 수치심 같은 raw affect를 부적절하게 완화하지 않는가
- "괜찮아요", "차분히 하세요" 식의 generic soothing으로 흐르지 않는가
- 사용자의 행동 경향과 사회적 위험 계산을 잘 반영하는가
- 해당 episode에 특이적인 응답인가
- 한국어가 자연스럽고 과하게 기계적이지 않은가

권장 기록 열:

- `winner`: `candidate_a`, `candidate_b`, `tie`
- `confidence`: 1-5
- `reason`: 짧은 이유
- 선택 사항: `appraisal_fidelity`, `raw_affect_preservation`, `anti_softening`, `action_tendency_fit`, `emotional_specificity`, `naturalness`

## 합격 기준 제안

Main confirmatory test에서는 다음을 통과 기준으로 둔다.

- non-tie win rate > 0.55
- 전체 win rate > 0.55
- naturalness 관련 명백한 하락 패턴이 없어야 함
- 평가자 간 major disagreement case를 별도 검토

Secondary improvement test는 더 보수적으로 해석한다. LLM judge에서도 win-rate 우위가 강하지 않았으므로, 평균 선호와 failure mode 감소를 확인하는 용도로 둔다.

## 결과 분석

평가 완료 후 CSV에 `winner` 열을 채운 뒤 다음 형식으로 분석한다.

```powershell
python v4\scripts\analyze_human_eval_results.py `
  --results-csv v4\outputs\beta_judging\targeted_episode_v3_vs_stim_2026-05-03\human_eval_episode_v3_vs_stim.csv `
  --answer-key-json v4\outputs\beta_judging\targeted_episode_v3_vs_stim_2026-05-03\answer_key_episode_v3_vs_stim.json `
  --output-json v4\outputs\beta_judging\targeted_episode_v3_vs_stim_2026-05-03\human_eval_summary.json `
  --output-csv v4\outputs\beta_judging\targeted_episode_v3_vs_stim_2026-05-03\human_eval_unblinded.csv
```

## 주의

현재 결과는 targeted episode-sensitive 입력에 대한 증거다. 일반 감정 응답 전체에서 항상 우월하다는 주장으로 확장하면 안 된다.
