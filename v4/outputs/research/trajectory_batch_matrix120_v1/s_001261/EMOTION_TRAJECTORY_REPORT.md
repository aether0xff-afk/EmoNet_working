# Emotion Trajectory Report

## Input

내 지인들은 내가 힘들어 할 때마다 날 도와줘. 정말 고마워. [SEP] 지인들이 힘들 때 다가와 줘서 기쁘시겠어요. [SEP] 가정형편이 어려워서 그런지 내가 티를 내지 않게 돼. 그것도 나름 고역이네. [SEP] 가정형편이 어려워서 힘들어하는 티를 내지 않으시군요. 어떻게 해야 힘들어하지 않을 수 있을까요? [SEP] 잠깐 휴식을 하고 주변 사람들에게 고맙다고 표현해야겠어. [SEP] 즐거운 휴식이 되었으면 좋겠어요.

## Trajectory Summary

- trajectory_pattern: escalation_to_fatigue_shift
- phase_count: 4
- phase_sequence: dormant -> ignition -> persistence -> fatigue_shift
- peak_alarm_tick: 16
- peak_fatigue_tick: 83
- peak_conflict_tick: 0
- dominant_global_signal: 추동/접근

## Phase Segments

- dormant (tick 0-2, duration 3) | dominant_signal=추동/접근 | top_emotion=무기력/철수 (0.3000)
- ignition (tick 3-5, duration 3) | dominant_signal=완충/억제 | top_emotion=무기력/철수 (0.4089)
- persistence (tick 6-13, duration 8) | dominant_signal=추동/접근 | top_emotion=소진/탈진 (0.4271)
- fatigue_shift (tick 14-127, duration 114) | dominant_signal=추동/접근 | top_emotion=소진/탈진 (0.5209)
