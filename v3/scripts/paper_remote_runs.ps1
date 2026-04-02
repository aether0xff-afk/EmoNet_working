param(
    [ValidateSet("e2e", "buildsubset", "labelsubset", "fitsubset", "predictsubset", "label500", "fit500", "predict500")]
    [string]$Task,
    [string]$PythonExe = "python",
    [string]$BaseUrl = "http://127.0.0.1:11434/v1",
    [string]$ModelName = "gpt-oss:20b",
    [int]$SubsetSize = 2000,
    [int]$FlushEvery = 10
)

$ErrorActionPreference = "Stop"

if ($Task -in @("label500", "fit500", "predict500")) {
    $SubsetSize = 500
}

$subsetCsv = if ($SubsetSize -eq 500) { ".\outputs\llm\llm_subset.csv" } else { ".\outputs\llm\llm_subset_$($SubsetSize).csv" }
$labeledCsv = ".\outputs\llm\llm_subset_labeled_$($SubsetSize)_ollama.csv"
$decoderPath = ".\artifacts\z_to_s_decoder_$($SubsetSize).npz"
$predCsv = ".\outputs\z\out_z_training_with_s_pred_$($SubsetSize).csv"

function Invoke-CheckedPython {
    param(
        [string[]]$Args
    )

    & $PythonExe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE: $PythonExe $($Args -join ' ')"
    }
}

switch ($Task) {
    "e2e" {
        Invoke-CheckedPython @(
            "-m", "emonet.cli", "e2e-check",
            "--text", "지금 너무 예민하고 피곤해.",
            "--zs-model-path", ".\artifacts\z_to_s_decoder.npz",
            "--base-url", $BaseUrl,
            "--model-name", $ModelName,
            "--report-json", ".\outputs\validation\e2e_check_report_success.json",
            "--output-csv", ".\outputs\validation\e2e_check_runs_success.csv",
            "--log-jsonl", ".\outputs\validation\e2e_check_runs_success.jsonl"
        )
    }
    "buildsubset" {
        Invoke-CheckedPython @(
            "-m", "emonet.cli", "build-llm-subset",
            "--input-csv", ".\outputs\z\out_z_training.csv",
            "--output-csv", $subsetCsv,
            "--target-size", "$SubsetSize",
            "--label-column", "label",
            "--seed", "42"
        )
    }
    "labelsubset" {
        Invoke-CheckedPython @(
            "-m", "emonet.cli", "label-local",
            "--input-csv", $subsetCsv,
            "--output-csv", $labeledCsv,
            "--base-url", $BaseUrl,
            "--model-name", $ModelName,
            "--block-size", "8",
            "--style-dim", "32",
            "--generation-temperature", "0.4",
            "--rating-temperature", "0.0",
            "--max-retries", "4",
            "--timeout-sec", "180",
            "--keep-threshold", "0.18",
            "--flush-every", "$FlushEvery",
            "--resume",
            "--keep-failures"
        )
    }
    "fitsubset" {
        Invoke-CheckedPython @(
            "-m", "emonet.cli", "fit-zs-regressor",
            "--input-csv", $labeledCsv,
            "--model-path", $decoderPath,
            "--val-ratio", "0.1"
        )
    }
    "predictsubset" {
        Invoke-CheckedPython @(
            "-m", "emonet.cli", "predict-s",
            "--input-csv", ".\outputs\z\out_z_training.csv",
            "--output-csv", $predCsv,
            "--model-path", $decoderPath
        )
    }
    "label500" {
        & $PSCommandPath -Task labelsubset -PythonExe $PythonExe -BaseUrl $BaseUrl -ModelName $ModelName -SubsetSize 500 -FlushEvery $FlushEvery
    }
    "fit500" {
        & $PSCommandPath -Task fitsubset -PythonExe $PythonExe -BaseUrl $BaseUrl -ModelName $ModelName -SubsetSize 500 -FlushEvery $FlushEvery
    }
    "predict500" {
        & $PSCommandPath -Task predictsubset -PythonExe $PythonExe -BaseUrl $BaseUrl -ModelName $ModelName -SubsetSize 500 -FlushEvery $FlushEvery
    }
}
