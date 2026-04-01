param(
    [string]$PythonExe = "python",
    [string]$BaseUrl = "http://127.0.0.1:11434/v1",
    [string]$ModelName = "gpt-oss:20b",
    [int]$SubsetSize = 2000,
    [int]$FlushEvery = 10
)

$ErrorActionPreference = "Stop"

$subsetCsv = if ($SubsetSize -eq 500) { ".\outputs\llm\llm_subset.csv" } else { ".\outputs\llm\llm_subset_$($SubsetSize).csv" }
$labeledCsv = ".\outputs\llm\llm_subset_labeled_$($SubsetSize)_ollama.csv"
$decoderPath = ".\artifacts\z_to_s_decoder_$($SubsetSize).npz"
$predCsv = ".\outputs\z\out_z_training_with_s_pred_$($SubsetSize).csv"
$metricsPath = ".\outputs\paper\paper_metrics_snapshot_remote_$($SubsetSize).json"

Write-Host "[1/6] End-to-end success run"
& $PythonExe -m emonet.cli e2e-check `
    --text "지금 너무 예민하고 피곤해." `
    --zs-model-path .\artifacts\z_to_s_decoder.npz `
    --base-url $BaseUrl `
    --model-name $ModelName `
    --report-json .\outputs\validation\e2e_check_report_success.json `
    --output-csv .\outputs\validation\e2e_check_runs_success.csv `
    --log-jsonl .\outputs\validation\e2e_check_runs_success.jsonl

Write-Host "[2/6] Build balanced subset ($SubsetSize rows target)"
& $PythonExe -m emonet.cli build-llm-subset `
    --input-csv .\outputs\z\out_z_training.csv `
    --output-csv $subsetCsv `
    --target-size $SubsetSize `
    --label-column label `
    --seed 42

Write-Host "[3/6] Label subset rows with resumable checkpoints"
& $PythonExe -m emonet.cli label-local `
    --input-csv $subsetCsv `
    --output-csv $labeledCsv `
    --base-url $BaseUrl `
    --model-name $ModelName `
    --block-size 8 `
    --style-dim 32 `
    --generation-temperature 0.4 `
    --rating-temperature 0.0 `
    --max-retries 4 `
    --timeout-sec 180 `
    --keep-threshold 0.18 `
    --flush-every $FlushEvery `
    --resume `
    --keep-failures

Write-Host "[4/6] Fit $SubsetSize-row z-to-s decoder"
& $PythonExe -m emonet.cli fit-zs-regressor `
    --input-csv $labeledCsv `
    --model-path $decoderPath `
    --val-ratio 0.1

Write-Host "[5/6] Predict s for full z training set"
& $PythonExe -m emonet.cli predict-s `
    --input-csv .\outputs\z\out_z_training.csv `
    --output-csv $predCsv `
    --model-path $decoderPath

Write-Host "[6/6] Recompute paper metrics snapshot"
& $PythonExe .\scripts\paper_metrics.py `
    --output-json $metricsPath `
    --llm-subset-csv $subsetCsv `
    --labeled-csv $labeledCsv `
    --labeled-summary-key "labeled_$($SubsetSize)_ollama"

Write-Host "[done] Artifacts"
Get-ChildItem .\outputs\paper,.\outputs\validation,.\outputs\llm,.\artifacts,.\outputs\z `
    -File `
    | Where-Object {
        $_.Name -in @(
            (Split-Path $metricsPath -Leaf),
            "e2e_check_report_success.json",
            "e2e_check_runs_success.csv",
            "e2e_check_runs_success.jsonl",
            (Split-Path $labeledCsv -Leaf),
            (Split-Path $decoderPath -Leaf),
            (Split-Path $predCsv -Leaf),
            (Split-Path $subsetCsv -Leaf)
        )
    } `
    | Select-Object FullName, Length, LastWriteTime `
    | Format-Table -AutoSize
