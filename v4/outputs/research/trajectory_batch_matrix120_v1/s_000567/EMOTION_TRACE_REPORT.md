# Emotion Trace Report

## Input

나 많이 긴장되고 애간장이 탈 것 같아. [SEP] 왜 그러신가요? 어떤것 때문에 그렇게 안절부절하시는 거예요? [SEP] 내일이 구월 모의고사를 치르는 날이거든. 그동안 열심히 공부했는데 공부한 만큼 점수가 잘 나오지 않을까봐 너무 불안해. [SEP] 열심히 준비한 만큼 점수가 안 나올까봐 불안하신가 보군요. [SEP] 응. 이번에는 반드시 잘 나와야 해. [SEP] 열심히 노력한 만큼 좋은 점수가 나왔으면 좋겠네요.

## Raw Trace Summary

- ticks_run: 75
- termination_reason: stable_convergence
- dominant_branch_len: 73
- persistence_ratio: 0.9733
- saturation_ratio: 0.6678
- dominant_global_signal: 공세적 긴장

## Raw Signal Means

- 추동/접근: 0.2488 (낮음)
- 완충/억제: 0.2307 (낮음)
- 경계/날카로움: 0.5719 (높음)
- 피로/둔화: 0.1843 (낮음)

## Candidate Emotions

- 예민함/신경과민: 0.5540 (높음)
- 방어적 경계: 0.5249 (중간)
- 짜증/분노압: 0.4885 (중간)

## Top Active Nodes

- node 234 [excitatory] 추동/접근 | activity_ticks=71, k_sum=150322.21, k_mean=2117.21
- node 195 [excitatory] 공세적 긴장 | activity_ticks=69, k_sum=148564.94, k_mean=2153.12
- node 115 [excitatory] 공세적 긴장 | activity_ticks=71, k_sum=136419.97, k_mean=1921.41
- node 191 [inhibitory] 완충/억제 | activity_ticks=69, k_sum=131649.06, k_mean=1907.96
- node 113 [inhibitory] 완충/억제 | activity_ticks=69, k_sum=119091.53, k_mean=1725.96
- node 190 [modulatory] 추동/접근 | activity_ticks=71, k_sum=118715.69, k_mean=1672.05
- node 55 [inhibitory] 완충/억제 | activity_ticks=71, k_sum=109226.23, k_mean=1538.40
- node 147 [excitatory] 추동/접근 | activity_ticks=71, k_sum=107488.63, k_mean=1513.92
- node 130 [excitatory] 공세적 긴장 | activity_ticks=71, k_sum=106504.48, k_mean=1500.06
- node 224 [inhibitory] 완충/억제 | activity_ticks=70, k_sum=105957.04, k_mean=1513.67
