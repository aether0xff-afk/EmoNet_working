# Branch Trace Analysis Report

- baseline: `baseline`
- compared_configs: `['baseline', 'random:k_threshold_base=0.82;k_remem_base=0.95;k_decay=0.97;refractory_ticks=3;input_signal_clip=1.0;recent_activity_decay=0.3;hysteresis_threshold_gain=0.0;hysteresis_remem_gain=0.0;hysteresis_k_bonus=0.0;memory_decay=0.98;memory_k_mix=0.0;state_base_stim_mix=0.05', 'random:k_threshold_base=0.82;k_remem_base=1.05;k_decay=0.97;refractory_ticks=3;input_signal_clip=1.0;recent_activity_decay=0.2;hysteresis_threshold_gain=0.03;hysteresis_remem_gain=0.0;hysteresis_k_bonus=0.0;memory_decay=0.98;memory_k_mix=0.0;state_base_stim_mix=0.05', 'random:k_threshold_base=0.9;k_remem_base=0.95;k_decay=0.97;refractory_ticks=3;input_signal_clip=1.0;recent_activity_decay=0.4;hysteresis_threshold_gain=0.03;hysteresis_remem_gain=0.0;hysteresis_k_bonus=0.02;memory_decay=0.98;memory_k_mix=0.1;state_base_stim_mix=0.05']`

## Config Comparison

### baseline

- balanced_score: `30.7500`
- constraint_penalty: `1.668333`
- constraint_failures: `hit_max_ticks_ratio>0.8;mean_first_active_tick>20.0;late_ignition_ratio_ge_15>0.4`
- mean_branch_len: `106.9500`
- len1_ratio: `0.0000`
- hit_max_ticks_ratio: `1.0000`
- mean_first_active_tick: `20.8667`
- late_ignition_ratio_ge_15: `0.9500`
- mean_active_nodes: `206.8333`
- mean_edges_fired: `1676.1388`
- p10/p50/p90 activity ticks: `33.0` / `75.0` / `117.0`

### random:k_threshold_base=0.82;k_remem_base=0.95;k_decay=0.97;refractory_ticks=3;input_signal_clip=1.0;recent_activity_decay=0.3;hysteresis_threshold_gain=0.0;hysteresis_remem_gain=0.0;hysteresis_k_bonus=0.0;memory_decay=0.98;memory_k_mix=0.0;state_base_stim_mix=0.05

- balanced_score: `30.0000`
- constraint_penalty: `1.955833`
- constraint_failures: `hit_max_ticks_ratio>0.8;mean_first_active_tick>20.0;late_ignition_ratio_ge_15>0.4`
- mean_branch_len: `103.8833`
- len1_ratio: `0.0000`
- hit_max_ticks_ratio: `1.0000`
- mean_first_active_tick: `24.1167`
- late_ignition_ratio_ge_15: `1.0000`
- mean_active_nodes: `67.9961`
- mean_edges_fired: `666.9150`
- p10/p50/p90 activity ticks: `36.0` / `76.0` / `117.0`

### random:k_threshold_base=0.82;k_remem_base=1.05;k_decay=0.97;refractory_ticks=3;input_signal_clip=1.0;recent_activity_decay=0.2;hysteresis_threshold_gain=0.03;hysteresis_remem_gain=0.0;hysteresis_k_bonus=0.0;memory_decay=0.98;memory_k_mix=0.0;state_base_stim_mix=0.05

- balanced_score: `30.0000`
- constraint_penalty: `1.955833`
- constraint_failures: `hit_max_ticks_ratio>0.8;mean_first_active_tick>20.0;late_ignition_ratio_ge_15>0.4`
- mean_branch_len: `103.8833`
- len1_ratio: `0.0000`
- hit_max_ticks_ratio: `1.0000`
- mean_first_active_tick: `24.1167`
- late_ignition_ratio_ge_15: `1.0000`
- mean_active_nodes: `68.0303`
- mean_edges_fired: `677.0533`
- p10/p50/p90 activity ticks: `35.0` / `76.0` / `117.0`

### random:k_threshold_base=0.9;k_remem_base=0.95;k_decay=0.97;refractory_ticks=3;input_signal_clip=1.0;recent_activity_decay=0.4;hysteresis_threshold_gain=0.03;hysteresis_remem_gain=0.0;hysteresis_k_bonus=0.02;memory_decay=0.98;memory_k_mix=0.1;state_base_stim_mix=0.05

- balanced_score: `30.8557`
- constraint_penalty: `2.125833`
- constraint_failures: `hit_max_ticks_ratio>0.8;mean_first_active_tick>20.0;late_ignition_ratio_ge_15>0.4`
- mean_branch_len: `100.4833`
- len1_ratio: `0.0000`
- hit_max_ticks_ratio: `1.0000`
- mean_first_active_tick: `27.5167`
- late_ignition_ratio_ge_15: `1.0000`
- mean_active_nodes: `65.7738`
- mean_edges_fired: `691.9499`
- p10/p50/p90 activity ticks: `38.0` / `78.0` / `118.0`

## Representative Samples

- sample_index=11 | category=delayed | first_active_tick=40.0 | dominant_branch_len=87.0
  - text: 치매가 약하게 오기 시작했어. 이젠 일상생활도 어려워지고 있어서 슬퍼. [SEP] 치매에 걸려 일상생활도 어려워져서 슬프시겠어요. [SEP] 자식들이 괜찮다고 치료받으면 된다지만 늦춰질 뿐 내 병을 심해질 테지. [SEP] 병이 심해질까 봐 더 걱정이시군요. [SEP] 이러면 자식들이 힘들어지는 건 아는데 이마저도 기억이 안 나겠지. [SEP] 도움이 되실 분이 곁에 계셨으면 좋겠어요.
- sample_index=49 | category=delayed | first_active_tick=29.0 | dominant_branch_len=99.0
  - text: 병원 복도에서 마주친 노인이 갑자기 나에게 구토를 했어. [SEP] 갑작스런 상황에 당황하셨겠어요. [SEP] 나도 저렇게 되는 게 아닐까 두려워. [SEP] 나도 병이 들까봐 걱정이 되는군요. [SEP] 이제부터라도 자기관리에 신경 써야겠어. [SEP] 앞으로 자기 관리에 더 힘쓸 생각이시군요.
- sample_index=41 | category=saturated | first_active_tick=12.0 | dominant_branch_len=116.0
  - text: 아이가 태어났는데 뭔가 문제가 있을 수 있다고 들었어. 너무 불안하고 두려워. [SEP] 좋지 않은 소식이네요. 많이 불편하시겠어요. [SEP] 아이에게 심각한 문제가 있으면 어떻게 할지 모르겠어. [SEP] 지금 이 기분에서 벗어나기 위해 할 수 있는 일이 무엇이 있을까요? [SEP] 의사와 얘기를 좀 더 해봐야겠어. [SEP] 마음이 조금이라도 더 편해지시면 좋겠어요.
- sample_index=42 | category=saturated | first_active_tick=14.0 | dominant_branch_len=114.0
  - text: 가난으로 고생했던 경험을 내 자식들에게까지 물려주게 될 것 같아서 겁이 나. [SEP] 자식들까지 고생을 하게 될까 봐 걱정이 되시는군요. [SEP] 좋은 것들만 하게 해주고 싶은데 자식들의 가난이 나 때문인 것 같아. [SEP] 지금 상황에 무엇을 할 수 있을까요? [SEP] 내 아이들이 조금이라도 더 행복해질 수 있는 방법이 무엇일지 고민해봐야겠어. [SEP] 아드님과의 대화가 잘 풀리길 바랄게요.
- sample_index=38 | category=short_branch | first_active_tick=27.0 | dominant_branch_len=101.0
  - text: 나 입사 면접에 또 떨어졌어. [SEP] 이번에도 입사 면접에 떨어지셔서 많이 속상하시겠군요. [SEP] 내 친구는 이번에 면접 합격했다는데 나 너무 괴로워. [SEP] 친구와는 다르게 합격을 못 하셔서 괴로우시군요. [SEP] 나는 왜 이리 취업을 못 할까? 너무 힘들어. [SEP] 계속되는 취업준비에 많이 괴로우시겠어요.
- sample_index=1 | category=short_branch | first_active_tick=26.0 | dominant_branch_len=102.0
  - text: 내 성과를 모두 가로챘던 직장 동료가 넘어져서 많이 다쳤대. 너무 통쾌해. [SEP] 회사 동료가 성과를 가로채서 많이 속상하셨군요. [SEP] 솔직히 벌 받은 거라고 생각해. 나는 그것보다 훨씬 많이 힘들었거든. [SEP] 어떻게 하면 기분이 좀 더 나아질 수 있을까요? [SEP] 그때의 기억은 잊으려고 노력하고 앞으로 당분간은 내 일을 더 열심히 해야겠어. [SEP] 딸과 대화하면서 기분이 나아졌으면 좋겠어요.

## Figures

- `config_mean_active_nodes.svg`
- `config_mean_edges_fired.svg`
- `config_activity_ratio.svg`
- `sample11_delayed_active_nodes.svg`
- `sample11_delayed_edges_fired.svg`
- `sample49_delayed_active_nodes.svg`
- `sample49_delayed_edges_fired.svg`
- `sample41_saturated_active_nodes.svg`
- `sample41_saturated_edges_fired.svg`
- `sample42_saturated_active_nodes.svg`
- `sample42_saturated_edges_fired.svg`
- `sample38_short_branch_active_nodes.svg`
- `sample38_short_branch_edges_fired.svg`
- `sample1_short_branch_active_nodes.svg`
- `sample1_short_branch_edges_fired.svg`
