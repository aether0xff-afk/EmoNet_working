# Emotion Trace Report

## Input

내가 심한 교통사고가 났을 때 어머니는 밤새 나를 간호해주었어. 어머니께 정말 감사해. [SEP] 어머니가 밤새 간호해주셨군요. 고마운 마음이 드시겠어요. [SEP] 나는 어머니께 해준 것도 없는데 미안하고 감사한 마음이 커. 지금이라도 효도하고 싶어. [SEP] 어떻게 하면 어머니에게 좋은 효도를 할 수 있을까요? [SEP] 집안일을 도와드리고 요리도 내가 직접 해드려야겠어. [SEP] 어머니에게 좋은 도움이 되었으면 좋겠어요.

## Raw Trace Summary

- ticks_run: 57
- termination_reason: stable_convergence
- dominant_branch_len: 55
- persistence_ratio: 0.9649
- saturation_ratio: 0.8398
- dominant_global_signal: 추동/접근

## Raw Signal Means

- 추동/접근: 0.5988 (높음)
- 완충/억제: 0.2391 (낮음)
- 경계/날카로움: 0.5167 (중간)
- 피로/둔화: 0.1907 (낮음)

## Candidate Emotions

- 예민함/신경과민: 0.6088 (높음)
- 짜증/분노압: 0.5712 (높음)
- 방어적 경계: 0.5017 (중간)

## Top Active Nodes

- node 78 [excitatory] 추동/접근 | activity_ticks=53, k_sum=25975.24, k_mean=490.10
- node 23 [excitatory] 추동/접근 | activity_ticks=53, k_sum=24257.63, k_mean=457.69
- node 14 [excitatory] 추동/접근 | activity_ticks=53, k_sum=23979.23, k_mean=452.44
- node 178 [excitatory] 추동/접근 | activity_ticks=54, k_sum=23583.06, k_mean=436.72
- node 66 [excitatory] 공세적 긴장 | activity_ticks=53, k_sum=23363.10, k_mean=440.81
- node 110 [inhibitory] 완충/억제 | activity_ticks=53, k_sum=22960.31, k_mean=433.21
- node 234 [excitatory] 추동/접근 | activity_ticks=52, k_sum=22758.31, k_mean=437.66
- node 179 [excitatory] 공세적 긴장 | activity_ticks=51, k_sum=22579.06, k_mean=442.73
- node 56 [excitatory] 추동/접근 | activity_ticks=54, k_sum=22351.66, k_mean=413.92
- node 83 [excitatory] 공세적 긴장 | activity_ticks=53, k_sum=22351.38, k_mean=421.72
