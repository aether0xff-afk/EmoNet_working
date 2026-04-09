# Emotion Trace Report

## Input

박 과장이 나한테 모욕적인 말을 해서 나도 화가 나서 불만을 말했어. [SEP] 모욕적인 말을 들어서 화가 많이 나고 속상하셨겠어요. [SEP] 맞아. 그런데 박 과장한테 나도 불만을 토로하니까 내일 회사에 가서 눈치가 보일 것 같아. [SEP] 회사 생활을 잘 이어가기 위해서 어떻게 하면 좋을까요? [SEP] 나는 모욕적인 언사에 대해서 정당하게 불만을 토로한 것이니 눈치 보지 말고 당당하게 있어야겠어. [SEP] 실력이 늘어서 나이가 많아도 계속 일할 수 있길 바라요.

## Raw Trace Summary

- ticks_run: 54
- termination_reason: stable_convergence
- dominant_branch_len: 52
- persistence_ratio: 0.9630
- saturation_ratio: 0.6874
- dominant_global_signal: 공세적 긴장

## Raw Signal Means

- 추동/접근: 0.2628 (낮음)
- 완충/억제: 0.2257 (낮음)
- 경계/날카로움: 0.5748 (높음)
- 피로/둔화: 0.2017 (낮음)

## Candidate Emotions

- 예민함/신경과민: 0.5593 (높음)
- 방어적 경계: 0.5224 (중간)
- 짜증/분노압: 0.4961 (중간)

## Top Active Nodes

- node 0 [excitatory] 공세적 긴장 | activity_ticks=49, k_sum=19649.30, k_mean=401.01
- node 138 [excitatory] 추동/접근 | activity_ticks=48, k_sum=18464.95, k_mean=384.69
- node 245 [excitatory] 추동/접근 | activity_ticks=49, k_sum=16733.76, k_mean=341.51
- node 64 [excitatory] 추동/접근 | activity_ticks=50, k_sum=15958.56, k_mean=319.17
- node 221 [modulatory] 공세적 긴장 | activity_ticks=51, k_sum=15752.19, k_mean=308.87
- node 94 [modulatory] 공세적 긴장 | activity_ticks=50, k_sum=15636.41, k_mean=312.73
- node 67 [inhibitory] 완충/억제 | activity_ticks=50, k_sum=15099.22, k_mean=301.98
- node 109 [excitatory] 추동/접근 | activity_ticks=49, k_sum=14498.71, k_mean=295.89
- node 149 [inhibitory] 완충/억제 | activity_ticks=49, k_sum=14463.55, k_mean=295.17
- node 172 [excitatory] 공세적 긴장 | activity_ticks=48, k_sum=14252.27, k_mean=296.92
