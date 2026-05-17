# Style Bias Audit

- input_csv: `outputs\research\style_relabel_v1\out_z_training_learned_extended40_calref_v1_style_relabel_v1.csv`
- rows: `1800`
- style_profile: `extended40`

## Keep Column Comparison

| keep column | rows | soft mean | negative raw mean | edge mean | top shifted axes |
|---|---:|---:|---:|---:|---|
| keep_sample | 1717 | 0.7846 | 0.0258 | 0.1241 | hostility=0.0093, playfulness=0.0146, volatility=0.0169, resentment=0.0214, metaphoricity=0.0220 |

## Focus Axes

### keep_sample
- softness: 0.8908
- calmness: 0.8719
- cooperativeness: 0.9020
- positivity: 0.8592
- warmth: 0.7351
- trust: 0.4486
- sharpness: 0.0794
- tension: 0.1252
- hostility: 0.0093
- resentment: 0.0214
- despair: 0.0319
- volatility: 0.0169
- fearfulness: 0.0421
- shame: 0.0331
- buckets: `{"mixed": 1514, "rare_raw": 111, "edgy": 92}`
- keep reasons: `{"consistent_nonsoft": 1536, "rare_affect_rescue": 181}`
