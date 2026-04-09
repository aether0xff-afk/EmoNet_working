# Emotion Trace Report

## Input

의사에게 무리를 해서라도 수술을 하면 앞을 볼 수 있냐고 물었어. [SEP] 의사에게 그 이야기를 할 때 어떤 감정이 들었나요? [SEP] 앞을 보는 건 어렵다는 이야기를 들을까 봐 불안했어. [SEP] 앞을 보지 못한다는 진단을 받을까 봐 걱정되셨군요. 지금 불안함에서 벗어나기 위해 어떤 것을 할 수 있을까요? [SEP] 가만히 있는 것보다는 운동하면서 체력을 좋게 만들어야겠어. [SEP] 운동으로 체력을 키우려고 하시는군요.

## Raw Trace Summary

- ticks_run: 118
- termination_reason: stable_convergence
- dominant_branch_len: 111
- persistence_ratio: 0.9407
- saturation_ratio: 0.8883
- dominant_global_signal: 공세적 긴장

## Raw Signal Means

- 추동/접근: 0.5790 (높음)
- 완충/억제: 0.2456 (낮음)
- 경계/날카로움: 0.5896 (높음)
- 피로/둔화: 0.2064 (낮음)

## Candidate Emotions

- 예민함/신경과민: 0.6455 (높음)
- 짜증/분노압: 0.5909 (높음)
- 방어적 경계: 0.5352 (중간)

## Top Active Nodes

- node 253 [excitatory] 공세적 긴장 | activity_ticks=109, k_sum=6480880.51, k_mean=59457.62
- node 5 [excitatory] 추동/접근 | activity_ticks=108, k_sum=6432191.93, k_mean=59557.33
- node 142 [excitatory] 추동/접근 | activity_ticks=108, k_sum=6129498.24, k_mean=56754.61
- node 203 [excitatory] 추동/접근 | activity_ticks=108, k_sum=5790598.34, k_mean=53616.65
- node 21 [modulatory] 공세적 긴장 | activity_ticks=108, k_sum=5703629.67, k_mean=52811.39
- node 38 [modulatory] 공세적 긴장 | activity_ticks=109, k_sum=5691102.43, k_mean=52211.95
- node 178 [excitatory] 추동/접근 | activity_ticks=109, k_sum=5628865.49, k_mean=51640.97
- node 54 [excitatory] 공세적 긴장 | activity_ticks=109, k_sum=5588372.99, k_mean=51269.48
- node 155 [modulatory] 공세적 긴장 | activity_ticks=108, k_sum=5560648.97, k_mean=51487.49
- node 246 [excitatory] 추동/접근 | activity_ticks=109, k_sum=5558846.08, k_mean=50998.59
