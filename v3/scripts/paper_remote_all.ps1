param(
    [string]$PythonExe = "python",
    [string]$BaseUrl = "http://127.0.0.1:11434/v1",
    [string]$ModelName = "gpt-oss:20b"
)

$ErrorActionPreference = "Stop"

Write-Host "[1/6] End-to-end success run"
& $PythonExe -m emonet.cli e2e-check `
    --text "지금 너무 예민하고 피곤해." `
    --zs-model-path .\artifacts\z_to_s_decoder.npz `
    --base-url $BaseUrl `
    --model-name $ModelName `
    --report-json .\outputs\validation\e2e_check_report_success.json `
    --output-csv .\outputs\validation\e2e_check_runs_success.csv `
    --log-jsonl .\outputs\validation\e2e_check_runs_success.jsonl

Write-Host "[2/6] Label 500 subset rows"
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

Write-Host "[3/6] Fit 500-row z-to-s decoder"
& $PythonExe -m emonet.cli fit-zs-regressor `
    --input-csv .\outputs\llm\llm_subset_labeled_500_ollama.csv `
    --model-path .\artifacts\z_to_s_decoder_500.npz `
    --val-ratio 0.1

Write-Host "[4/6] Predict s for full z training set"
& $PythonExe -m emonet.cli predict-s `
    --input-csv .\outputs\z\out_z_training.csv `
    --output-csv .\outputs\z\out_z_training_with_s_pred_500.csv `
    --model-path .\artifacts\z_to_s_decoder_500.npz

Write-Host "[5/6] Recompute paper metrics snapshot"
& $PythonExe .\scripts\paper_metrics.py `
    --output-json .\outputs\paper\paper_metrics_snapshot_remote.json

Write-Host "[6/6] Done"
Get-ChildItem .\outputs\paper,.\outputs\validation,.\outputs\llm,.\artifacts,.\outputs\z `
    -File `
    | Where-Object {
        $_.Name -in @(
            "paper_metrics_snapshot_remote.json",
            "e2e_check_report_success.json",
            "e2e_check_runs_success.csv",
            "e2e_check_runs_success.jsonl",
            "llm_subset_labeled_500_ollama.csv",
            "z_to_s_decoder_500.npz",
            "out_z_training_with_s_pred_500.csv"
        )
    } `
    | Select-Object FullName, Length, LastWriteTime `
    | Format-Table -AutoSize
