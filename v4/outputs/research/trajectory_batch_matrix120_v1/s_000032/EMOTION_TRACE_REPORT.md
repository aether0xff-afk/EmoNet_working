# Emotion Trace Report

## Input

우리 부서에는 운동 잘하는 사람들이 모여 있어서 체육대회에서 우승할 것 같아. [SEP] 체육대회에서 우승할 것 같다니 좋으시겠어요. [SEP] 사람들이 운동을 다 잘해서 나도 더 열심히 해야겠다는 생각을 했어. [SEP] 앞으로 어떻게 하실 계획이세요? [SEP] 열심히 운동해서 우승까지 하고 싶어. [SEP] 운동으로 우승까지 원하시는군요.

## Raw Trace Summary

- ticks_run: 45
- termination_reason: stable_convergence
- dominant_branch_len: 43
- persistence_ratio: 0.9556
- saturation_ratio: 0.6234
- dominant_global_signal: 추동/접근

## Raw Signal Means

- 추동/접근: 0.5303 (중간)
- 완충/억제: 0.2691 (낮음)
- 경계/날카로움: 0.4986 (중간)
- 피로/둔화: 0.1996 (낮음)

## Candidate Emotions

- 예민함/신경과민: 0.5590 (높음)
- 짜증/분노압: 0.5369 (중간)
- 방어적 경계: 0.5030 (중간)

## Top Active Nodes

- node 158 [inhibitory] 완충/억제 | activity_ticks=40, k_sum=6165.08, k_mean=154.13
- node 233 [excitatory] 추동/접근 | activity_ticks=41, k_sum=5908.34, k_mean=144.11
- node 184 [excitatory] 공세적 긴장 | activity_ticks=40, k_sum=5888.39, k_mean=147.21
- node 17 [inhibitory] 수축/둔화 | activity_ticks=39, k_sum=5751.94, k_mean=147.49
- node 9 [inhibitory] 완충/억제 | activity_ticks=41, k_sum=5751.31, k_mean=140.28
- node 55 [inhibitory] 완충/억제 | activity_ticks=41, k_sum=5394.45, k_mean=131.57
- node 31 [excitatory] 추동/접근 | activity_ticks=41, k_sum=5260.17, k_mean=128.30
- node 188 [inhibitory] 완충/억제 | activity_ticks=39, k_sum=4915.15, k_mean=126.03
- node 238 [inhibitory] 완충/억제 | activity_ticks=40, k_sum=4894.32, k_mean=122.36
- node 189 [inhibitory] 완충/억제 | activity_ticks=38, k_sum=4731.82, k_mean=124.52
