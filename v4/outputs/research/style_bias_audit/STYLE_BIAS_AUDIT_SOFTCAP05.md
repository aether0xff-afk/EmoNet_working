# Style Bias Audit

- input_csv: `outputs\research\style_bias_audit\out_z_training_learned_extended40_calref_v1_rebalanced_softcap05.csv`
- rows: `1800`
- style_profile: `extended40`

## Keep Column Comparison

| keep column | rows | soft mean | negative raw mean | edge mean | top shifted axes |
|---|---:|---:|---:|---:|---|
| keep_sample | 1717 | 0.8148 | 0.0032 | 0.1039 | hostility=0.0003, resentment=0.0003, shame=0.0017, volatility=0.0022, despair=0.0044 |
| keep_stylebias_softcap05 | 1717 | 0.8148 | 0.0032 | 0.1039 | hostility=0.0003, resentment=0.0003, shame=0.0017, volatility=0.0022, despair=0.0044 |

## Focus Axes

### keep_sample
- softness: 0.9276
- calmness: 0.9132
- cooperativeness: 0.9202
- positivity: 0.9051
- warmth: 0.7596
- trust: 0.4632
- sharpness: 0.0641
- tension: 0.0962
- hostility: 0.0003
- resentment: 0.0003
- despair: 0.0044
- volatility: 0.0022
- fearfulness: 0.0100
- shame: 0.0017
- buckets: `{"mixed": 1514, "rare_raw": 111, "edgy": 92}`
- keep reasons: `{"consistent_nonsoft": 1536, "rare_affect_rescue": 181}`

### keep_stylebias_softcap05
- softness: 0.9276
- calmness: 0.9132
- cooperativeness: 0.9202
- positivity: 0.9051
- warmth: 0.7596
- trust: 0.4632
- sharpness: 0.0641
- tension: 0.0962
- hostility: 0.0003
- resentment: 0.0003
- despair: 0.0044
- volatility: 0.0022
- fearfulness: 0.0100
- shame: 0.0017
- buckets: `{"mixed": 1514, "rare_raw": 111, "edgy": 92}`
- keep reasons: `{"consistent_nonsoft": 1536, "rare_affect_rescue": 181}`
