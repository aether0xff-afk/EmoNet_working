# Emotion Trace Report

## Input

은퇴하고 가입한 자전거 동호회에서 라이딩을 하다 크게 다쳤는데 수술비가 걱정이야. [SEP] 갑자기 큰 사고를 당해 걱정이 많으시겠어요. [SEP] 몸도 몸이지만 보험도 적용이 안 돼 수술비가 많이 나올 것 같아. [SEP] 수술비가 근심거리이군요. 어떻게 하는 게 좋을까요? [SEP] 별 수 있나. 수술 받고 치료해야지. [SEP] 수술 잘 끝내고 빨리 완쾌하시길 바라요.

## Raw Trace Summary

- ticks_run: 75
- termination_reason: stable_convergence
- dominant_branch_len: 73
- persistence_ratio: 0.9733
- saturation_ratio: 0.7045
- dominant_global_signal: 공세적 긴장

## Raw Signal Means

- 추동/접근: 0.2662 (낮음)
- 완충/억제: 0.2263 (낮음)
- 경계/날카로움: 0.5753 (높음)
- 피로/둔화: 0.1882 (낮음)

## Candidate Emotions

- 예민함/신경과민: 0.5635 (높음)
- 방어적 경계: 0.5287 (중간)
- 짜증/분노압: 0.4920 (중간)

## Top Active Nodes

- node 96 [excitatory] 추동/접근 | activity_ticks=71, k_sum=170689.76, k_mean=2404.08
- node 217 [excitatory] 추동/접근 | activity_ticks=72, k_sum=135406.17, k_mean=1880.64
- node 4 [modulatory] 추동/접근 | activity_ticks=71, k_sum=129166.70, k_mean=1819.25
- node 138 [excitatory] 추동/접근 | activity_ticks=70, k_sum=126089.93, k_mean=1801.28
- node 181 [inhibitory] 완충/억제 | activity_ticks=71, k_sum=125190.03, k_mean=1763.24
- node 103 [excitatory] 추동/접근 | activity_ticks=72, k_sum=124693.36, k_mean=1731.85
- node 185 [inhibitory] 완충/억제 | activity_ticks=71, k_sum=120278.35, k_mean=1694.06
- node 254 [excitatory] 공세적 긴장 | activity_ticks=70, k_sum=119215.35, k_mean=1703.08
- node 91 [modulatory] 추동/접근 | activity_ticks=70, k_sum=114054.50, k_mean=1629.35
- node 250 [excitatory] 추동/접근 | activity_ticks=70, k_sum=111855.58, k_mean=1597.94
