param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$PaperDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir = Join-Path $PaperDir "build"

if (-not (Test-Path -LiteralPath $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

if ($Clean) {
    Get-ChildItem -LiteralPath $BuildDir -Force | Remove-Item -Force -Recurse
}

Push-Location $PaperDir
try {
    & xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
    & xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
}
finally {
    Pop-Location
}
