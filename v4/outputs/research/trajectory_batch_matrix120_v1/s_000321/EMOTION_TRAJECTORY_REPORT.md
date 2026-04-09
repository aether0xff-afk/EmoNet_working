# Emotion Trajectory Report

## Input

수술하는 의사가 되고 싶었는데 손을 심하게 다쳐버렸어. 영영 낫지 못할 것 같아서 슬퍼. [SEP] 의사가 되고 싶은데 손을 다쳐서 너무 속상하시겠어요. [SEP] 아픈 사람들을 수술로 치료해주고 싶었는데 손을 다쳐서 그러지 못할 것 같아. 절망스러워. [SEP] 많이 힘드시겠어요. 앞으로 어떻게 하는 것이 좋을까요? [SEP] 치료에 전념하면서 선생님과도 진로 상담을 해보고 싶어. [SEP] 꾸준한 치료와 진로 상담으로 마음을 잘 극복하길 바라요.

## Trajectory Summary

- trajectory_pattern: mixed
- phase_count: 4
- phase_sequence: dormant -> ignition -> escalation -> persistence
- peak_alarm_tick: 102
- peak_fatigue_tick: 9
- peak_conflict_tick: 0
- dominant_global_signal: 공세적 긴장

## Phase Segments

- dormant (tick 0-1, duration 2) | dominant_signal=추동/접근 | top_emotion=무기력/철수 (0.3000)
- ignition (tick 2-4, duration 3) | dominant_signal=경계/날카로움 | top_emotion=짜증/분노압 (0.4059)
- escalation (tick 5-5, duration 1) | dominant_signal=경계/날카로움 | top_emotion=방어적 경계 (0.3812)
- persistence (tick 6-127, duration 122) | dominant_signal=경계/날카로움 | top_emotion=예민함/신경과민 (0.5772)
