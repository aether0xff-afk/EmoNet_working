# Output Layout

Generated files are organized under `outputs/`.

## Z Exports

- `outputs/z/out_z_training.csv`
- `outputs/z/out_z_sample.csv`

## LLM Labeling

- `outputs/llm/llm_subset.csv`
- `outputs/llm/llm_subset_prompts.jsonl`
- `outputs/llm/llm_subset_labeled_test.csv`
- `outputs/llm/llm_subset_labeled_50.csv`
- `outputs/llm/llm_subset_labeled_50_v2.csv`
- `outputs/llm/llm_subset_labeled_50_ollama.csv`

## Model Artifacts

- `artifacts/ridge_stim_encoder.joblib`
- `artifacts/zs/z_to_s_decoder.npz`

## Recommended Commands

```powershell
python -m emonet.cli build-llm-subset `
  --input-csv .\outputs\z\out_z_training.csv `
  --output-csv .\outputs\llm\llm_subset.csv `
  --prompt-jsonl .\outputs\llm\llm_subset_prompts.jsonl `
  --target-size 500
```

```powershell
python -m emonet.cli label-local `
  --input-csv .\outputs\llm\llm_subset.csv `
  --output-csv .\outputs\llm\llm_subset_labeled_50_ollama.csv `
  --base-url "http://127.0.0.1:11434/v1" `
  --model-name "gpt-oss:20b" `
  --limit 50 `
  --block-size 8 `
  --generation-temperature 0.4 `
  --rating-temperature 0.0 `
  --max-retries 4 `
  --timeout-sec 180 `
  --keep-threshold 0.18 `
  --keep-failures
```

```powershell
python -m emonet.cli fit-zs-regressor `
  --input-csv .\outputs\llm\llm_subset_labeled_50_ollama.csv `
  --model-path .\artifacts\zs\z_to_s_decoder.npz `
  --val-ratio 0.1
```

```powershell
python -m emonet.cli predict-s `
  --input-csv .\outputs\z\out_z_training.csv `
  --output-csv .\outputs\z\out_z_training_with_s_pred.csv `
  --model-path .\artifacts\zs\z_to_s_decoder.npz
```
