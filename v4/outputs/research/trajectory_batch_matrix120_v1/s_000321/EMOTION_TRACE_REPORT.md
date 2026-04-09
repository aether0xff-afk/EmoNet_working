# Emotion Trace Report

## Input

수술하는 의사가 되고 싶었는데 손을 심하게 다쳐버렸어. 영영 낫지 못할 것 같아서 슬퍼. [SEP] 의사가 되고 싶은데 손을 다쳐서 너무 속상하시겠어요. [SEP] 아픈 사람들을 수술로 치료해주고 싶었는데 손을 다쳐서 그러지 못할 것 같아. 절망스러워. [SEP] 많이 힘드시겠어요. 앞으로 어떻게 하는 것이 좋을까요? [SEP] 치료에 전념하면서 선생님과도 진로 상담을 해보고 싶어. [SEP] 꾸준한 치료와 진로 상담으로 마음을 잘 극복하길 바라요.

## Raw Trace Summary

- ticks_run: 128
- termination_reason: max_ticks
- dominant_branch_len: 126
- persistence_ratio: 0.9844
- saturation_ratio: 0.6940
- dominant_global_signal: 공세적 긴장

## Raw Signal Means

- 추동/접근: 0.2526 (낮음)
- 완충/억제: 0.2183 (낮음)
- 경계/날카로움: 0.5867 (높음)
- 피로/둔화: 0.1755 (낮음)

## Candidate Emotions

- 예민함/신경과민: 0.5684 (높음)
- 방어적 경계: 0.5288 (중간)
- 짜증/분노압: 0.4982 (중간)

## Top Active Nodes

- node 145 [excitatory] 추동/접근 | activity_ticks=121, k_sum=22521873.37, k_mean=186131.18
- node 221 [modulatory] 공세적 긴장 | activity_ticks=122, k_sum=19551451.90, k_mean=160257.80
- node 96 [excitatory] 추동/접근 | activity_ticks=124, k_sum=19159052.47, k_mean=154508.49
- node 187 [modulatory] 경계/날카로움 | activity_ticks=122, k_sum=18855928.11, k_mean=154556.79
- node 222 [excitatory] 추동/접근 | activity_ticks=119, k_sum=18557227.05, k_mean=155943.08
- node 80 [modulatory] 공세적 긴장 | activity_ticks=122, k_sum=17148189.72, k_mean=140558.93
- node 212 [inhibitory] 완충/억제 | activity_ticks=122, k_sum=16849132.60, k_mean=138107.64
- node 160 [excitatory] 공세적 긴장 | activity_ticks=123, k_sum=15202437.34, k_mean=123597.05
- node 25 [excitatory] 추동/접근 | activity_ticks=123, k_sum=14944838.29, k_mean=121502.75
- node 207 [inhibitory] 완충/억제 | activity_ticks=122, k_sum=14857494.51, k_mean=121782.74
