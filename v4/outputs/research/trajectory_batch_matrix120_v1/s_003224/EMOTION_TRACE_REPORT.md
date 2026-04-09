# Emotion Trace Report

## Input

내가 배달하다가 오토바이 사고를 당했는데 다리를 못 쓸 거 같다고 해. [SEP] 사고를 당해서 다리를 못 쓸 것 같은 상황이라 아주 슬프시겠어요. [SEP] 좌절할 가족들을 생각하면 마음이 너무 아파서 이 상황을 어떻게 해야 할지 모르겠어. [SEP] 지금 상황에서 어떻게 하는 것이 가장 좋을까요? [SEP] 솔직하게 지금의 상황을 얘기하고 가족들의 도움을 받아야겠어. [SEP] 가족들에게 솔직히 말씀드려서 지금의 어려움이 한결 나아지길 바라요.

## Raw Trace Summary

- ticks_run: 68
- termination_reason: stable_convergence
- dominant_branch_len: 66
- persistence_ratio: 0.9706
- saturation_ratio: 0.8268
- dominant_global_signal: 공세적 긴장

## Raw Signal Means

- 추동/접근: 0.2518 (낮음)
- 완충/억제: 0.2398 (낮음)
- 경계/날카로움: 0.5683 (높음)
- 피로/둔화: 0.1952 (낮음)

## Candidate Emotions

- 예민함/신경과민: 0.5672 (높음)
- 방어적 경계: 0.5319 (중간)
- 짜증/분노압: 0.4804 (중간)

## Top Active Nodes

- node 103 [excitatory] 추동/접근 | activity_ticks=64, k_sum=85032.75, k_mean=1328.64
- node 217 [excitatory] 추동/접근 | activity_ticks=63, k_sum=78489.25, k_mean=1245.86
- node 96 [excitatory] 추동/접근 | activity_ticks=62, k_sum=72580.73, k_mean=1170.66
- node 138 [excitatory] 추동/접근 | activity_ticks=64, k_sum=71669.50, k_mean=1119.84
- node 86 [excitatory] 공세적 긴장 | activity_ticks=63, k_sum=70532.86, k_mean=1119.57
- node 29 [inhibitory] 완충/억제 | activity_ticks=64, k_sum=70349.63, k_mean=1099.21
- node 185 [inhibitory] 완충/억제 | activity_ticks=64, k_sum=70033.34, k_mean=1094.27
- node 254 [excitatory] 공세적 긴장 | activity_ticks=63, k_sum=68911.51, k_mean=1093.83
- node 36 [inhibitory] 완충/억제 | activity_ticks=62, k_sum=68681.43, k_mean=1107.77
- node 58 [excitatory] 공세적 긴장 | activity_ticks=64, k_sum=68486.81, k_mean=1070.11
