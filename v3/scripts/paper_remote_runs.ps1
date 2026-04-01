param(
    [ValidateSet("e2e", "label500", "fit500", "predict500")]
    [string]$Task,
    [string]$PythonExe = "python",
    [string]$BaseUrl = "http://127.0.0.1:11434/v1",
    [string]$ModelName = "gpt-oss:20b"
)

$ErrorActionPreference = "Stop"

switch ($Task) {
    "e2e" {
        & $PythonExe -m emonet.cli e2e-check `
            --text "지금 너무 예민하고 피곤해." `
            --zs-model-path .\artifacts\z_to_s_decoder.npz `
            --base-url $BaseUrl `
            --model-name $ModelName `
            --report-json .\outputs\validation\e2e_check_report_success.json `
            --output-csv .\outputs\validation\e2e_check_runs_success.csv `
            --log-jsonl .\outputs\validation\e2e_check_runs_success.jsonl
    }
    "label500" {
        & $PythonExe -m emonet.cli label-local `
            --input-csv .\outputs\llm\llm_subset.csv `
            --output-csv .\outputs\llm\llm_subset_labeled_500_ollama.csv `
            --base-url $BaseUrl `
            --model-name $ModelName `
            --limit 500 `
            --block-size 8 `
            --style-dim 32 `
            --generation-temperature 0.4 `
            --rating-temperature 0.0 `
            --max-retries 4 `
            --timeout-sec 180 `
            --keep-threshold 0.18 `
            --keep-failures
    }
    "fit500" {
        & $PythonExe -m emonet.cli fit-zs-regressor `
            --input-csv .\outputs\llm\llm_subset_labeled_500_ollama.csv `
            --model-path .\artifacts\z_to_s_decoder_500.npz `
            --val-ratio 0.1
    }
    "predict500" {
        & $PythonExe -m emonet.cli predict-s `
            --input-csv .\outputs\z\out_z_training.csv `
            --output-csv .\outputs\z\out_z_training_with_s_pred_500.csv `
            --model-path .\artifacts\z_to_s_decoder_500.npz
    }
}
