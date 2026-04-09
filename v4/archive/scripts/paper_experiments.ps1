param(
    [ValidateSet("matrix", "evalpack", "all")]
    [string]$Task = "all",
    [string]$PythonExe = "python",
    [string]$BaseUrl = "http://127.0.0.1:11434/v1",
    [string]$ModelName = "gpt-oss:20b",
    [string]$InputCsv = ".\outputs\llm\llm_subset.csv",
    [string]$ZsModelPath = ".\artifacts\z_to_s_decoder.npz",
    [string]$Conditions = "direct,stim_only,emonet_full,emonet_no_summary,emonet_no_tags,emonet_vector_only",
    [int]$Limit = 100,
    [int]$SampleSize = 100
)

$ErrorActionPreference = "Stop"

$matrixCsv = ".\outputs\experiments\paper_matrix.csv"
$matrixSummary = ".\outputs\experiments\paper_matrix_summary.json"
$matrixJsonl = ".\outputs\experiments\paper_matrix.jsonl"
$evalCsv = ".\outputs\experiments\paper_human_eval.csv"
$evalKey = ".\outputs\experiments\paper_human_eval_key.json"
$evalGuide = ".\outputs\experiments\paper_human_eval_instructions.md"

function Invoke-CheckedPython {
    param(
        [string[]]$Args
    )

    & $PythonExe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE: $PythonExe $($Args -join ' ')"
    }
}

if ($Task -in @("matrix", "all")) {
    Invoke-CheckedPython @(
        ".\scripts\experiment_matrix.py",
        "--input-csv", $InputCsv,
        "--output-csv", $matrixCsv,
        "--summary-json", $matrixSummary,
        "--log-jsonl", $matrixJsonl,
        "--zs-model-path", $ZsModelPath,
        "--base-url", $BaseUrl,
        "--model-name", $ModelName,
        "--conditions", $Conditions,
        "--limit", "$Limit",
        "--resume"
    )
}

if ($Task -in @("evalpack", "all")) {
    Invoke-CheckedPython @(
        ".\scripts\prepare_human_eval.py",
        "--input-csv", $matrixCsv,
        "--output-csv", $evalCsv,
        "--answer-key-json", $evalKey,
        "--instructions-md", $evalGuide,
        "--conditions", $Conditions,
        "--sample-size", "$SampleSize"
    )
}

Get-ChildItem .\outputs\experiments -File `
    | Select-Object FullName, Length, LastWriteTime `
    | Format-Table -AutoSize
