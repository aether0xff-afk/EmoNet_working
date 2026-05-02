# Paired Superiority Analysis

Baseline condition: `stim_only`

## Mean Total Comparisons

| condition | paired_n | delta_mean | delta_median | wins | ties | losses | win_rate | bootstrap_ci_low | bootstrap_ci_high | sign_test_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| episode_trace | 113 | -0.081416 | 0.0 | 47 | 15 | 51 | 0.415929 | -0.226549 | 0.056637 | 0.76203622 |
| raw_trace | 113 | -0.180531 | 0.0 | 32 | 26 | 55 | 0.283186 | -0.311504 | -0.049558 | 0.01782755 |
| emonet_full | 113 | -0.281416 | 0.0 | 42 | 15 | 56 | 0.371681 | -0.470796 | -0.097345 | 0.18884672 |
| hybrid_episode | 111 | -0.336937 | -0.2 | 36 | 13 | 62 | 0.324324 | -0.527928 | -0.145946 | 0.01117454 |
| direct | 113 | -0.033628 | 0.0 | 40 | 26 | 47 | 0.353982 | -0.152212 | 0.083186 | 0.52029159 |

## Metric-Level Comparisons

| condition | metric | paired_n | delta_mean | wins | ties | losses | win_rate | bootstrap_ci_low | bootstrap_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| episode_trace | mean_total | 113 | -0.081416 | 47 | 15 | 51 | 0.415929 | -0.226549 | 0.056637 |
| episode_trace | content_fit | 113 | 0.141593 | 30 | 67 | 16 | 0.265487 | -0.017699 | 0.300885 |
| episode_trace | emotional_appropriateness | 113 | -0.070796 | 27 | 55 | 31 | 0.238938 | -0.256637 | 0.106195 |
| episode_trace | style_match | 113 | -0.132743 | 25 | 53 | 35 | 0.221239 | -0.300885 | 0.026549 |
| episode_trace | naturalness | 113 | -0.265487 | 14 | 64 | 35 | 0.123894 | -0.460398 | -0.070796 |
| episode_trace | overall_quality | 113 | -0.079646 | 26 | 60 | 27 | 0.230088 | -0.256637 | 0.088496 |
| raw_trace | mean_total | 113 | -0.180531 | 32 | 26 | 55 | 0.283186 | -0.311504 | -0.049558 |
| raw_trace | content_fit | 113 | 0.026549 | 23 | 71 | 19 | 0.20354 | -0.123894 | 0.176991 |
| raw_trace | emotional_appropriateness | 113 | -0.123894 | 22 | 58 | 33 | 0.19469 | -0.300885 | 0.044248 |
| raw_trace | style_match | 113 | -0.327434 | 13 | 56 | 44 | 0.115044 | -0.469027 | -0.185841 |
| raw_trace | naturalness | 113 | -0.283186 | 17 | 53 | 43 | 0.150442 | -0.486726 | -0.088496 |
| raw_trace | overall_quality | 113 | -0.19469 | 21 | 53 | 39 | 0.185841 | -0.362832 | -0.035398 |
| emonet_full | mean_total | 113 | -0.281416 | 42 | 15 | 56 | 0.371681 | -0.470796 | -0.097345 |
| emonet_full | content_fit | 113 | -0.221239 | 26 | 55 | 32 | 0.230088 | -0.442478 | 0.0 |
| emonet_full | emotional_appropriateness | 113 | -0.230088 | 28 | 46 | 39 | 0.247788 | -0.451327 | -0.00885 |
| emonet_full | style_match | 113 | 0.053097 | 34 | 55 | 24 | 0.300885 | -0.123894 | 0.221239 |
| emonet_full | naturalness | 113 | -0.716814 | 13 | 51 | 49 | 0.115044 | -0.982522 | -0.451327 |
| emonet_full | overall_quality | 113 | -0.292035 | 23 | 52 | 38 | 0.20354 | -0.513274 | -0.070796 |
| hybrid_episode | mean_total | 111 | -0.336937 | 36 | 13 | 62 | 0.324324 | -0.527928 | -0.145946 |
| hybrid_episode | content_fit | 111 | -0.207207 | 30 | 47 | 34 | 0.27027 | -0.432432 | 0.018018 |
| hybrid_episode | emotional_appropriateness | 111 | -0.216216 | 25 | 50 | 36 | 0.225225 | -0.432432 | 0.000225 |
| hybrid_episode | style_match | 111 | 0.072072 | 31 | 54 | 26 | 0.279279 | -0.099099 | 0.243243 |
| hybrid_episode | naturalness | 111 | -0.945946 | 13 | 36 | 62 | 0.117117 | -1.225225 | -0.657658 |
| hybrid_episode | overall_quality | 111 | -0.387387 | 22 | 43 | 46 | 0.198198 | -0.621622 | -0.153153 |
| direct | mean_total | 113 | -0.033628 | 40 | 26 | 47 | 0.353982 | -0.152212 | 0.083186 |
| direct | content_fit | 113 | 0.00885 | 26 | 65 | 22 | 0.230088 | -0.150442 | 0.159292 |
| direct | emotional_appropriateness | 113 | -0.044248 | 23 | 60 | 30 | 0.20354 | -0.185841 | 0.097345 |
| direct | style_match | 113 | -0.061947 | 19 | 71 | 23 | 0.168142 | -0.19469 | 0.061947 |
| direct | naturalness | 113 | -0.044248 | 19 | 72 | 22 | 0.168142 | -0.221239 | 0.132743 |
| direct | overall_quality | 113 | -0.026549 | 24 | 63 | 26 | 0.212389 | -0.185841 | 0.132743 |

## Episode Subsets

| condition | subset_axis | subset_value | paired_n | delta_mean | wins | ties | losses | win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct | arousal | medium | 9 | 0.111111 | 4 | 2 | 3 | 0.444444 |
| direct | arousal | high | 102 | -0.029412 | 36 | 24 | 42 | 0.352941 |
| direct | arousal | low | 2 | -0.9 | 0 | 0 | 2 | 0.0 |
| direct | control_state | high | 8 | -0.025 | 5 | 1 | 2 | 0.625 |
| direct | control_state | mixed | 78 | -0.028205 | 25 | 20 | 33 | 0.320513 |
| direct | control_state | low | 27 | -0.051852 | 10 | 5 | 12 | 0.37037 |
| direct | social_orientation | defend | 37 | 0.059459 | 15 | 9 | 13 | 0.405405 |
| direct | social_orientation | approach | 43 | -0.055814 | 13 | 11 | 19 | 0.302326 |
| direct | social_orientation | withdraw | 11 | -0.072727 | 4 | 3 | 4 | 0.363636 |
| direct | social_orientation | mixed | 22 | -0.127273 | 8 | 3 | 11 | 0.363636 |
| direct | target | self | 7 | 0.228571 | 4 | 1 | 2 | 0.571429 |
| direct | target | other | 12 | 0.1 | 5 | 3 | 4 | 0.416667 |
| direct | target | mixed | 79 | -0.065823 | 28 | 18 | 33 | 0.35443 |
| direct | target | situation | 15 | -0.093333 | 3 | 4 | 8 | 0.2 |
| direct | valence | negative | 85 | -0.004706 | 31 | 20 | 34 | 0.364706 |
| direct | valence | positive | 12 | -0.066667 | 5 | 2 | 5 | 0.416667 |
| direct | valence | mixed | 16 | -0.1625 | 4 | 4 | 8 | 0.25 |
| emonet_full | arousal | high | 102 | -0.252941 | 38 | 14 | 50 | 0.372549 |
| emonet_full | arousal | low | 2 | -0.4 | 0 | 0 | 2 | 0.0 |
| emonet_full | arousal | medium | 9 | -0.577778 | 4 | 1 | 4 | 0.444444 |
| emonet_full | control_state | low | 27 | -0.237037 | 10 | 3 | 14 | 0.37037 |
| emonet_full | control_state | mixed | 78 | -0.238462 | 31 | 10 | 37 | 0.397436 |
| emonet_full | control_state | high | 8 | -0.85 | 1 | 2 | 5 | 0.125 |
| emonet_full | social_orientation | defend | 36 | -0.033333 | 17 | 7 | 12 | 0.472222 |
| emonet_full | social_orientation | withdraw | 11 | -0.054545 | 6 | 1 | 4 | 0.545455 |
| emonet_full | social_orientation | mixed | 22 | -0.218182 | 7 | 1 | 14 | 0.318182 |
| emonet_full | social_orientation | approach | 44 | -0.572727 | 12 | 6 | 26 | 0.272727 |
| emonet_full | target | self | 7 | 0.371429 | 5 | 1 | 1 | 0.714286 |
| emonet_full | target | other | 12 | 0.05 | 7 | 1 | 4 | 0.583333 |
| emonet_full | target | situation | 14 | -0.157143 | 3 | 4 | 7 | 0.214286 |
| emonet_full | target | mixed | 80 | -0.41 | 27 | 9 | 44 | 0.3375 |
| emonet_full | valence | negative | 84 | -0.128571 | 35 | 11 | 38 | 0.416667 |
| emonet_full | valence | mixed | 17 | -0.682353 | 5 | 1 | 11 | 0.294118 |
| emonet_full | valence | positive | 12 | -0.783333 | 2 | 3 | 7 | 0.166667 |
| episode_trace | arousal | medium | 9 | 0.088889 | 4 | 2 | 3 | 0.444444 |
| episode_trace | arousal | high | 102 | -0.086275 | 43 | 13 | 46 | 0.421569 |
| episode_trace | arousal | low | 2 | -0.6 | 0 | 0 | 2 | 0.0 |
| episode_trace | control_state | low | 27 | -0.074074 | 12 | 2 | 13 | 0.444444 |
| episode_trace | control_state | mixed | 78 | -0.074359 | 32 | 12 | 34 | 0.410256 |
| episode_trace | control_state | high | 8 | -0.175 | 3 | 1 | 4 | 0.375 |
| episode_trace | social_orientation | mixed | 22 | 0.072727 | 11 | 1 | 10 | 0.5 |
| episode_trace | social_orientation | withdraw | 11 | 0.072727 | 4 | 1 | 6 | 0.363636 |
| episode_trace | social_orientation | approach | 44 | -0.086364 | 17 | 9 | 18 | 0.386364 |
| episode_trace | social_orientation | defend | 36 | -0.216667 | 15 | 4 | 17 | 0.416667 |
| episode_trace | target | other | 11 | 0.109091 | 8 | 1 | 2 | 0.727273 |
| episode_trace | target | mixed | 80 | -0.02 | 33 | 10 | 37 | 0.4125 |
| episode_trace | target | self | 7 | -0.4 | 3 | 1 | 3 | 0.428571 |
| episode_trace | target | situation | 15 | -0.4 | 3 | 3 | 9 | 0.2 |
| episode_trace | valence | negative | 84 | -0.054762 | 37 | 10 | 37 | 0.440476 |
| episode_trace | valence | mixed | 17 | -0.117647 | 7 | 2 | 8 | 0.411765 |
| episode_trace | valence | positive | 12 | -0.216667 | 3 | 3 | 6 | 0.25 |
| hybrid_episode | arousal | low | 2 | 0.0 | 1 | 0 | 1 | 0.5 |
| hybrid_episode | arousal | medium | 9 | -0.222222 | 3 | 1 | 5 | 0.333333 |
| hybrid_episode | arousal | high | 100 | -0.354 | 32 | 12 | 56 | 0.32 |
| hybrid_episode | control_state | mixed | 77 | -0.280519 | 27 | 9 | 41 | 0.350649 |
| hybrid_episode | control_state | low | 26 | -0.346154 | 8 | 2 | 16 | 0.307692 |
| hybrid_episode | control_state | high | 8 | -0.85 | 1 | 2 | 5 | 0.125 |
| hybrid_episode | social_orientation | withdraw | 11 | -0.018182 | 4 | 1 | 6 | 0.363636 |
| hybrid_episode | social_orientation | defend | 35 | -0.28 | 12 | 3 | 20 | 0.342857 |
| hybrid_episode | social_orientation | mixed | 21 | -0.352381 | 9 | 1 | 11 | 0.428571 |
| hybrid_episode | social_orientation | approach | 44 | -0.454545 | 11 | 8 | 25 | 0.25 |
| hybrid_episode | target | other | 11 | -0.036364 | 4 | 1 | 6 | 0.363636 |
| hybrid_episode | target | self | 7 | -0.057143 | 2 | 0 | 5 | 0.285714 |
| hybrid_episode | target | mixed | 79 | -0.379747 | 28 | 10 | 41 | 0.35443 |
| hybrid_episode | target | situation | 14 | -0.471429 | 2 | 2 | 10 | 0.142857 |
| hybrid_episode | valence | negative | 83 | -0.231325 | 29 | 9 | 45 | 0.349398 |
| hybrid_episode | valence | mixed | 16 | -0.55 | 5 | 3 | 8 | 0.3125 |
| hybrid_episode | valence | positive | 12 | -0.783333 | 2 | 1 | 9 | 0.166667 |
| raw_trace | arousal | high | 102 | -0.168627 | 29 | 25 | 48 | 0.284314 |
| raw_trace | arousal | medium | 9 | -0.266667 | 3 | 0 | 6 | 0.333333 |
| raw_trace | arousal | low | 2 | -0.4 | 0 | 1 | 1 | 0.0 |
| raw_trace | control_state | low | 27 | -0.044444 | 11 | 4 | 12 | 0.407407 |
| raw_trace | control_state | mixed | 78 | -0.161538 | 21 | 20 | 37 | 0.269231 |
| raw_trace | control_state | high | 8 | -0.825 | 0 | 2 | 6 | 0.0 |
| raw_trace | social_orientation | withdraw | 11 | 0.163636 | 5 | 2 | 4 | 0.454545 |
| raw_trace | social_orientation | defend | 36 | -0.116667 | 10 | 11 | 15 | 0.277778 |
| raw_trace | social_orientation | mixed | 22 | -0.127273 | 9 | 3 | 10 | 0.409091 |
| raw_trace | social_orientation | approach | 44 | -0.345455 | 8 | 10 | 26 | 0.181818 |
| raw_trace | target | other | 11 | 0.254545 | 6 | 2 | 3 | 0.545455 |
| raw_trace | target | self | 7 | 0.114286 | 4 | 2 | 1 | 0.571429 |
| raw_trace | target | mixed | 80 | -0.22 | 20 | 17 | 43 | 0.25 |
| raw_trace | target | situation | 15 | -0.426667 | 2 | 5 | 8 | 0.133333 |
| raw_trace | valence | negative | 84 | -0.088095 | 28 | 20 | 36 | 0.333333 |
| raw_trace | valence | mixed | 17 | -0.364706 | 3 | 4 | 10 | 0.176471 |
| raw_trace | valence | positive | 12 | -0.566667 | 1 | 2 | 9 | 0.083333 |

## Largest Wins

| condition | record_id | delta_mean_total | episode_label | valence | arousal |
| --- | --- | --- | --- | --- | --- |
| episode_trace | s_001621 | 2.0000000000000004 | 걱정 기반의 조심스러운 확인 충동 | negative | high |
| hybrid_episode | s_001146 | 2.0 | 방어적 자살사고 경계 고착 | negative | high |
| raw_trace | s_002540 | 1.7999999999999998 | 노화 해석에 묶인 피로성 경계 | negative | medium |
| emonet_full | s_001146 | 1.7999999999999998 | 방어적 자살사고 경계 고착 | negative | high |
| emonet_full | s_002456 | 1.6 | 명예퍄손 위협에 대한 경계적 반격 | negative | high |
| hybrid_episode | s_000889 | 1.6 | 죄책감 기반의 초조한 만회 추동 | mixed | high |
| direct | s_000070 | 1.6 | 불공정 인식에 묶인 공격적 환멸 | negative | high |
| hybrid_episode | s_000166 | 1.4 | 배제 위협에 대한 경계성 위축 | negative | high |

## Largest Losses

| condition | record_id | delta_mean_total | episode_label | valence | arousal |
| --- | --- | --- | --- | --- | --- |
| emonet_full | s_003590 | -3.8 | 추억 재점화형 애정 고양 | positive | high |
| emonet_full | s_001320 | -3.8 | 배신-모욕 기반의 공세적 복수 긴장 | negative | high |
| emonet_full | s_001822 | -3.4000000000000004 | 감사-미안함의 보답 추동 | mixed | high |
| emonet_full | s_000149 | -3.2 | 신뢰 기반 기대 고양 | positive | medium |
| episode_trace | s_003782 | -3.0 | 자기부담화된 미안함과 회복 강박 | negative | high |
| hybrid_episode | s_000149 | -2.8000000000000003 | 신뢰 기반 기대 고양 | positive | medium |
| hybrid_episode | s_001185 | -2.8000000000000003 | 고각성 접근형 기대-만족 | positive | high |
| hybrid_episode | s_003022 | -2.8 | 자기경계가 섞인 공세적 긴장 | negative | high |

## Artifacts

- overall CSV: `C:\Users\remote\Documents\GitHub\EmoNet_working\v4\outputs\experiments\paired_superiority_episode_v2\paired_overall.csv`
- subset CSV: `C:\Users\remote\Documents\GitHub\EmoNet_working\v4\outputs\experiments\paired_superiority_episode_v2\paired_subsets.csv`
- examples CSV: `C:\Users\remote\Documents\GitHub\EmoNet_working\v4\outputs\experiments\paired_superiority_episode_v2\paired_examples.csv`
- summary JSON: `C:\Users\remote\Documents\GitHub\EmoNet_working\v4\outputs\experiments\paired_superiority_episode_v2\paired_summary.json`
