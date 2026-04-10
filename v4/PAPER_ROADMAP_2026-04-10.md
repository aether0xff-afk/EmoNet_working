# Paper Roadmap (2026-04-10)

## 목적

이 문서는 `v4` 기준으로 논문을 어떤 메시지로 정리할지, 지금 무엇을 주장할 수 있고 무엇은 아직 주장하면 안 되는지, 그리고 남은 실험을 어떤 우선순위로 처리할지를 한 곳에 고정하기 위한 로드맵이다.

현재 가장 중요한 목표는 "미해결 과제가 있으니 논문을 미루자"가 아니라, **이미 방어 가능한 메시지를 먼저 잠그고 그 위에 부족한 증거만 추가하는 것**이다.

## 1. 현재 논문의 한 줄 메시지

현재 데이터로 가장 방어 가능한 한 줄은 아래다.

> EmoNet은 branch collapse를 실질적으로 완화하고 raw trajectory를 더 풍부한 affect episode로 해석할 수 있게 만들었지만, 그 내부 품질이 아직 최종 generation superiority로 완전히 번역되지는 않는다.

이 문장을 벗어나면 과장 위험이 커진다.

## 2. 이 논문을 어떤 종류의 논문으로 써야 하는가

현재 버전은 "최종 성능 최고 모델" 논문보다는 아래 성격에 더 가깝다.

- 감정 생성 파이프라인을 단계별로 분해하고 진단 가능하게 만든 시스템 논문
- branch dynamics calibration과 trajectory interpretation을 중심에 둔 분석 논문
- end-to-end 품질 향상과 실패 원인을 함께 보여주는 honest paper

즉, 주력 메시지는 "우리가 모든 것을 이겼다"가 아니라 "어디가 실제 병목인지 이전보다 훨씬 좁히고 측정할 수 있게 되었다"여야 한다.

## 3. 지금 바로 주장 가능한 핵심 기여

### 3.1 Branch collapse recovery

- full export `51,628` rows 기준 dominant branch 평균 길이가 `1.0539 -> 70.4684`로 증가했다.
- `len1_ratio`는 `0.9734 -> 0.0948`로 감소했다.
- 따라서 branch-based intermediate representation은 이제 거의 trivial하지 않다.

### 3.2 Calibration-backed reference configuration

- reference config는 임의값이 아니라 calibration experiment로 선택되었다.
- current selected config는 `no_activity_ratio=0.05`, `len1_ratio=0.05`, `hit_max_ticks_ratio=0.25`, `mean_first_active_tick=2.40`, `mean_branch_len=79.78`를 만족한다.
- 따라서 "좋아 보여서 정한 하이퍼파라미터"가 아니라, 점화 실패와 collapse, 과도한 persistence 사이의 균형을 실험적으로 고른 설정이라고 말할 수 있다.

### 3.3 Raw trajectory -> episode interpretation

- heuristic top-emotion readout보다 `raw trajectory -> GPT-5.4 episode interpretation` 경로가 더 풍부한 episode-level 설명을 제공한다.
- 현재는 이 경로를 "정답 emotion 복원기"로 쓰면 안 되지만, trajectory를 사람이 읽을 수 있는 semantic episode로 번역하는 해석기로는 충분히 의미가 있다.

## 4. 지금 본문에 반드시 함께 써야 하는 한계

아래 한계는 숨기지 말고 main story에 포함하는 것이 맞다.

### 4.1 End-to-end 최고 성능은 아직 `stim_only`

- 최신 비교에서 `stim_only = 3.5404`
- `episode_trace = 3.4673`
- `emonet_full = 3.2478`

즉, EmoNet의 내부 구조가 개선되었다고 해서 최종 generation score가 자동으로 최고가 되지는 않았다.

### 4.2 Style target bias가 강하다

- `softness = 0.9276`
- `calmness = 0.9132`
- `cooperativeness = 0.9202`
- `positivity = 0.9051`
- 반대로 `hostility`, `resentment`, `despair`, `volatility`는 거의 0에 가깝다.

즉, 내부 affect signal이 좋아져도 supervision target 자체가 최종 표면을 다시 순화시킬 수 있다.

### 4.3 `z -> s` predictor는 아직 text baseline보다 약하다

- `text_tfidf = 0.114628`
- `structfix_learned_z64 = 0.117328`

따라서 현재 learned latent가 predictor superiority를 확보했다고 쓰면 안 된다.

### 4.4 Upper-tail saturation은 아직 남아 있다

- branch `p90=126`, `p95=126`, `max=126`

collapse는 크게 완화되었지만, 일부 샘플은 여전히 ceiling 부근까지 간다.

## 5. 현재 단계에서 금지해야 할 주장

아래 문장은 현재 자료로는 방어하기 어렵다.

- EmoNet이 인간처럼 감정을 느낀다
- episode interpreter가 ground-truth emotion을 정확히 복원한다
- current latent가 text baseline보다 우월하다
- episode conditioning이 전체 generation에서 최고다
- end-to-end 품질까지 포함해 EmoNet이 baseline을 전반적으로 이겼다

## 6. 추천 논문 서사

가장 안정적인 서사는 아래 순서다.

### 6.1 Problem framing

- 기존 감정 생성은 라벨 주입이나 surface prompt 중심이라 실패 원인을 내부 단계별로 추적하기 어렵다.
- EmoNet은 입력 -> dynamics -> branch/trajectory -> latent -> style -> surface로 분해한다.

### 6.2 Structural recovery

- 가장 먼저 branch collapse 문제를 보여준다.
- calibration과 structfix 이후 branch가 non-trivial trajectory가 되었다는 증거를 제시한다.

### 6.3 Interpretation layer

- 하지만 길어진 branch만으로는 사람이 이해하기 어렵다.
- 그래서 raw trajectory를 episode interpretation으로 변환하는 경로를 도입했다고 설명한다.

### 6.4 End-to-end reality check

- 내부 quality는 개선되었지만, 최종 generation에서는 아직 `stim_only`를 완전히 넘지 못한다.
- 이 차이를 style bias, surface softening, `episode -> response` translation bottleneck으로 해석한다.

### 6.5 Takeaway

- EmoNet의 현재 기여는 "최고 점수"보다 "감정 처리의 내부 병목을 측정 가능하게 만든 것"에 있다.

## 7. 본문 구조 초안

`PAPER_DRAFT_ko.md`는 아래 구조로 다듬는 것이 좋다.

1. Introduction
2. Related framing
3. EmoNet pipeline
4. Branch collapse diagnosis and calibration
5. Trajectory-to-episode interpretation
6. End-to-end generation comparison
7. Failure analysis
8. Limitations
9. Conclusion

각 섹션의 핵심 문장은 아래처럼 고정한다.

### 7.1 Introduction

- 감정 생성의 핵심 문제는 표현 품질 자체보다도 내부 감정 처리 실패를 분해해서 관찰하기 어렵다는 점이다.

### 7.2 Pipeline

- EmoNet은 감정 생성을 단일 prompt trick이 아니라 단계적 affect processing pipeline으로 모델링한다.

### 7.3 Calibration

- calibration 이전에는 branch가 거의 1-step에서 끝났고, 이후에는 긴 trajectory가 형성된다.

### 7.4 Interpretation

- raw trajectory는 heuristic emotion tag보다 더 풍부한 episode semantics를 담는다.

### 7.5 End-to-end results

- internal trace quality improvement는 generation quality improvement와 동일하지 않다.

### 7.6 Failure analysis

- 현재 주병목은 style bias와 surface softening이다.

## 8. 지금 바로 넣을 그림과 표

### 8.1 필수 그림

- `outputs/paper/refresh_2026-04-09_calref_v1/figures/dominant_branch_before_after.svg`
- `outputs/paper/refresh_2026-04-09_calref_v1/figures/dominant_branch_length_distribution_structfix.svg`
- `outputs/paper/refresh_2026-04-09_calref_v1/figures/predictor_mae_comparison_current.svg`
- `outputs/paper/refresh_2026-04-09_calref_v1/figures/style_bias_axes_extended40.svg`
- `outputs/paper/refresh_2026-04-09_calref_v1/figures/style_consistency_histogram_extended40.svg`

### 8.2 필수 표

- branch recovery summary table
- calibration-backed reference config table
- end-to-end generation comparison table
- episode interpretation qualitative case table 4~6개

## 9. 남은 과제 우선순위

논문을 위해 남은 과제는 "필수"와 "후속"으로 나눠야 한다.

### 9.1 필수

1. `episode_trace` / `hybrid_episode` 전체 matrix를 최신 judge 기준으로 다시 채점
2. 최신 score 기준으로 본문 표/그림 refresh
3. `episode_trace - stim_only` per-sample delta 분포 확인
4. bootstrap CI 또는 paired win-rate 중 최소 하나 추가
5. 대표 qualitative case 4~6개를 고정

### 9.2 있으면 강해지는 것

1. episode interpretation paraphrase consistency
2. causal ablation metric
3. felt-state와 response-style supervision 분리

### 9.3 지금 논문을 막을 정도는 아닌 것

1. `legacy_cli.py` 완전 분해
2. predictor를 baseline보다 확실히 넘기는 대규모 재설계
3. 완전한 positive-arousal regime 해결

즉, 코드 구조 리팩터링은 중요하지만 논문 작성 시작의 선행조건은 아니다.

## 10. 작성 순서 추천

가장 현실적인 작성 순서는 아래다.

1. Introduction / Problem / Contribution부터 먼저 잠근다.
2. Calibration과 branch recovery 결과를 먼저 표와 그림으로 고정한다.
3. trajectory interpretation qualitative section을 붙인다.
4. end-to-end result를 honest하게 정리한다.
5. limitation과 next-step을 분명히 쓴다.

즉, **성능이 더 좋아진 뒤에 쓰는 방식이 아니라, 이미 확보된 구조적 결과를 기준으로 먼저 쓰고 부족한 end-to-end 증거만 채우는 방식**이 맞다.

## 11. 오늘 기준 권장 작업

오늘 바로 해야 할 일은 아래 셋이다.

1. `PAPER_DRAFT_ko.md`의 초록과 결론을 현재 메시지에 맞게 다시 잠근다.
2. `episode_trace` / `hybrid_episode` 최신 채점을 돌려 표를 갱신한다.
3. qualitative case table에 넣을 대표 샘플을 4~6개 선정한다.

이 세 가지가 끝나면, 논문은 "준비 단계"가 아니라 이미 "초안 작성 단계"로 넘어간다.
