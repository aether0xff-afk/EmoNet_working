# v4 Completion Status

작성일: 2026-05-03

## 요약

v4는 앱/논문/평가 파이프라인의 active 작업선이며, 현재 연구적으로는 broad superiority가 아니라 targeted episode-fidelity claim으로 정리하는 것이 맞다.

완료된 핵심 결과는 `superiority_targeted_v1`이다. `episode_trace_v3`는 episode 정보가 필요한 targeted 입력에서 `stim_only`보다 강하게 우수했다.

## 완료된 것

- Targeted episode-sensitive set 80개 구성 완료
- `stim_only`, `episode_trace`, `episode_trace_v3` generation matrix 생성 완료
- Targeted superiority judge 설계 및 scoring 완료
- Paired bootstrap/win-rate 분석 완료
- 차트와 보고서 생성 완료
- Human blind A/B용 beta package 생성 완료

## 핵심 수치

### `episode_trace_v3` vs `stim_only`

- Paired n: 78
- `mean_total` delta: +1.8308
- Median delta: +2.0000
- Bootstrap 95% CI: [+1.5667, +2.0897]
- Win / Tie / Loss: 70 / 3 / 5
- Win rate: 0.8974
- Naturalness: 4.4000 vs 4.1410

### `episode_trace_v3` vs `episode_trace`

- Paired n: 77
- `mean_total` delta: +0.4519
- Bootstrap 95% CI: [+0.1403, +0.7558]
- Win / Tie / Loss: 41 / 6 / 30
- Win rate: 0.5325
- Sign test p: 0.2351

## 해석

현재 v4가 말할 수 있는 강한 주장은 다음이다.

> episode 정보가 필요한 targeted 감정 입력에서 `episode_trace_v3`는 appraisal fidelity, raw affect preservation, anti-softening, action tendency fit, emotional specificity 기준으로 `stim_only`보다 우수하다.

반대로 아직 말하면 안 되는 주장은 다음이다.

> EmoNet이 모든 일반 감정 응답 생성에서 `stim_only`보다 항상 우수하다.

## 남은 일

1. Human blind A/B를 실제 사람 평가로 수행한다.
2. `episode_trace_v3` vs `stim_only` 결과를 confirmatory claim으로 우선 확정한다.
3. `episode_trace_v3` vs `episode_trace`는 개선 신호로 보고하되, 사람 평가에서 재확인한다.
4. `target_other` bucket을 30개 이상으로 확장한다.
5. 사람 평가 통과 후 paper claim을 broad superiority에서 targeted episode-fidelity superiority로 수정한다.

## 산출물 위치

- LLM judge targeted report: `v4/outputs/experiments/superiority_targeted_v1/SUPERIORITY_TARGETED_REPORT.md`
- Main human A/B package: `v4/outputs/beta_judging/targeted_episode_v3_vs_stim_2026-05-03/`
- Secondary human A/B package: `v4/outputs/beta_judging/targeted_episode_v3_vs_episode_2026-05-03/`
