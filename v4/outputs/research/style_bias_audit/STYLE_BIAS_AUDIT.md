# Style Bias Audit

- input_csv: `outputs\z\out_z_training_learned_extended40_calref_v1.csv`
- rows: `1800`
- style_profile: `extended40`

## Keep Column Comparison

| keep column | rows | soft mean | negative raw mean | edge mean | top shifted axes |
|---|---:|---:|---:|---:|---|
| keep_sample | 1717 | 0.8148 | 0.0032 | 0.1039 | hostility=0.0003, resentment=0.0003, shame=0.0017, volatility=0.0022, despair=0.0044 |
| keep_sample_rebalanced | 1800 | 0.8198 | 0.0030 | 0.1049 | hostility=0.0003, resentment=0.0003, shame=0.0017, volatility=0.0021, despair=0.0042 |

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

### keep_sample_rebalanced
- softness: 0.9293
- calmness: 0.9172
- cooperativeness: 0.9239
- positivity: 0.9093
- warmth: 0.7618
- trust: 0.4775
- sharpness: 0.0631
- tension: 0.0960
- hostility: 0.0003
- resentment: 0.0003
- despair: 0.0042
- volatility: 0.0021
- fearfulness: 0.0096
- shame: 0.0017
- buckets: `{"mixed": 1514, "rare_raw": 111, "edgy": 92, "soft_safe": 83}`
- keep reasons: `{"consistent_nonsoft": 1536, "rare_affect_rescue": 181, "oversoft_trim": 83}`
