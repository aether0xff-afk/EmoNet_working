param(
    [ValidateSet("offline", "baselines", "all")]
    [string]$Task = "all",
    [string]$PythonExe = "python",
    [string]$BaseUrl = "http://127.0.0.1:11434/v1",
    [string]$ModelName = "gpt-oss:20b",
    [string]$InputCsv = ".\outputs\llm\llm_subset.csv",
    [string]$ZsModelPath = ".\artifacts\z_to_s_decoder.npz",
    [string]$Conditions = "direct,stim_only,emonet_full",
    [int]$Limit = 100,
    [int]$SampleLimit = 0
)

$ErrorActionPreference = "Stop"

function Invoke-CheckedPython {
    param(
        [string[]]$Args
    )

    & $PythonExe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $PythonExe $($Args -join ' ')"
    }
}

if ($Task -in @("offline", "all")) {
    $offlineArgs = @(
        ".\scripts\paper_offline_tables.py"
    )
    if ($SampleLimit -gt 0) {
        $offlineArgs += @("--sample-limit", "$SampleLimit")
    }
    Invoke-CheckedPython $offlineArgs
}

if ($Task -in @("baselines", "all")) {
    Invoke-CheckedPython @(
        ".\scripts\experiment_matrix.py",
        "--input-csv", $InputCsv,
        "--output-csv", ".\outputs\experiments\paper_matrix.csv",
        "--summary-json", ".\outputs\experiments\paper_matrix_summary.json",
        "--log-jsonl", ".\outputs\experiments\paper_matrix.jsonl",
        "--zs-model-path", $ZsModelPath,
        "--base-url", $BaseUrl,
        "--model-name", $ModelName,
        "--conditions", $Conditions,
        "--limit", "$Limit",
        "--resume"
    )

    Invoke-CheckedPython @(
        ".\scripts\score_experiment_matrix.py",
        "--input-csv", ".\outputs\experiments\paper_matrix.csv",
        "--output-csv", ".\outputs\experiments\paper_matrix_scored.csv",
        "--summary-csv", ".\outputs\paper\requested_tables\baseline_generation_table.csv",
        "--summary-json", ".\outputs\paper\requested_tables\baseline_generation_table.json",
        "--base-url", $BaseUrl,
        "--model-name", $ModelName,
        "--max-tokens", "600",
        "--max-retries", "4",
        "--keep-failures",
        "--resume"
    )
}

Get-ChildItem .\outputs\paper\requested_tables,.\outputs\experiments -File -ErrorAction SilentlyContinue `
    | Select-Object FullName, Length, LastWriteTime `
    | Sort-Object FullName `
    | Format-Table -AutoSize
