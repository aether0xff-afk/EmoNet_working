# Emotion Trace Report

## Input

매일 같은 옷 입는 게 이상해 보일까? [SEP] 지저분하지만 않다면 같은 옷이라도 상관없을 것 같아요. [SEP] 나도 다른 친구들처럼 꾸미고 싶은데 생활비 빼고 나면 양말 살 돈도 남질 않아. [SEP] 주변의 친구들이 옷보단 내면의 모습을 더욱 중요하게 볼 거 같습니다. [SEP] 정말 그랬으면 좋겠어. 나는 친구들에게 좋은 모습을 보여주려고 노력하고 있어. [SEP] 친구들도 그 모습을 보며 지내려고 할 거예요.

## Raw Trace Summary

- ticks_run: 55
- termination_reason: stable_convergence
- dominant_branch_len: 53
- persistence_ratio: 0.9636
- saturation_ratio: 0.8278
- dominant_global_signal: 공세적 긴장

## Raw Signal Means

- 추동/접근: 0.3336 (낮음)
- 완충/억제: 0.2491 (낮음)
- 경계/날카로움: 0.5552 (높음)
- 피로/둔화: 0.2056 (낮음)

## Candidate Emotions

- 예민함/신경과민: 0.5745 (높음)
- 방어적 경계: 0.5243 (중간)
- 짜증/분노압: 0.5015 (중간)

## Top Active Nodes

- node 202 [excitatory] 공세적 긴장 | activity_ticks=51, k_sum=24095.18, k_mean=472.45
- node 171 [excitatory] 추동/접근 | activity_ticks=52, k_sum=23731.84, k_mean=456.38
- node 208 [excitatory] 추동/접근 | activity_ticks=51, k_sum=22791.27, k_mean=446.89
- node 235 [modulatory] 피로성 경계 | activity_ticks=51, k_sum=22383.78, k_mean=438.90
- node 198 [excitatory] 추동/접근 | activity_ticks=51, k_sum=22158.03, k_mean=434.47
- node 109 [excitatory] 추동/접근 | activity_ticks=50, k_sum=22036.67, k_mean=440.73
- node 134 [excitatory] 추동/접근 | activity_ticks=52, k_sum=21949.52, k_mean=422.11
- node 23 [excitatory] 추동/접근 | activity_ticks=49, k_sum=21791.25, k_mean=444.72
- node 205 [inhibitory] 완충/억제 | activity_ticks=51, k_sum=21656.88, k_mean=424.64
- node 67 [inhibitory] 완충/억제 | activity_ticks=51, k_sum=21268.03, k_mean=417.02
