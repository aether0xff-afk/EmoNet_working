$ErrorActionPreference = "Stop"

$downloads = Join-Path $env:USERPROFILE "Downloads"
$orig = Get-ChildItem $downloads -Filter "EmoNet*.hwpx" |
  Where-Object { $_.Length -gt 2800000 -and $_.Name -notmatch "수정본|revision|hancom|settext|test" } |
  Sort-Object Length -Descending |
  Select-Object -First 1

if (-not $orig) {
  throw "원본 HWPX를 찾지 못했습니다."
}

$outAscii = Join-Path $downloads "EmoNet_preserve_edit.hwpx"
$outKorean = Join-Path $downloads "EmoNet_최종본_수정본.hwpx"
Remove-Item -LiteralPath $outAscii -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $outKorean -Force -ErrorAction SilentlyContinue

$hwp = New-Object -ComObject HWPFrame.HwpObject
$replacementPath = Join-Path (Get-Location) "tmp\hwp_replacements.json"

function Replace-AllText {
  param(
    [Parameter(Mandatory=$true)][string]$Find,
    [AllowEmptyString()][string]$Replace
  )
  $hwp.HAction.GetDefault("AllReplace", $hwp.HParameterSet.HFindReplace.HSet) | Out-Null
  $set = $hwp.HParameterSet.HFindReplace
  $set.Direction = $hwp.FindDir("AllDoc")
  $set.FindString = $Find
  $set.ReplaceString = $Replace
  $set.ReplaceMode = 1
  $set.IgnoreMessage = 1
  $set.FindType = 1
  $set.MatchCase = 0
  $set.AllWordForms = 0
  $set.SeveralWords = 0
  $set.UseWildCards = 0
  $set.WholeWordOnly = 0
  $hwp.HAction.Execute("AllReplace", $set.HSet) | Out-Null
}

function Append-Text {
  param([Parameter(Mandatory=$true)][string]$Text)
  $hwp.Run("MoveDocEnd") | Out-Null
  $hwp.Insert("`r`n" + $Text, 0, 0)
}

try {
  $hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule") | Out-Null
  try { $hwp.XHwpWindows.Item(0).Visible = $false } catch {}
  $opened = $hwp.Open($orig.FullName, "HWPX", "forceopen:true")
  if (-not $opened) {
    throw "한글 COM으로 원본을 열지 못했습니다: $($orig.FullName)"
  }

  $items = Get-Content -LiteralPath $replacementPath -Raw -Encoding UTF8 | ConvertFrom-Json
  foreach ($item in $items) {
    Replace-AllText ([string]$item.find) ([string]$item.replace)
  }

  # Append heatmap image without deleting existing figures, preserving original objects/tables.
  $img = Join-Path (Get-Location) "tmp\hwpx_revision_final\BinData\image9.PNG"
  if (Test-Path $img) {
    Append-Text "`r`n[수정] 그림 9 heatmap 이미지"
    $hwp.InsertPicture($img, $true, 0, $false, $false, 0, 0, 0) | Out-Null
  }

  $saved = $hwp.SaveAs($outAscii, "HWPX", "")
  if (-not $saved) {
    throw "SaveAs 실패"
  }
}
finally {
  try { $hwp.Quit() } catch {}
}

Copy-Item -LiteralPath $outAscii -Destination $outKorean -Force

# Verify it opens in Hancom.
Start-Sleep -Seconds 2
$verify = New-Object -ComObject HWPFrame.HwpObject
try {
  $ok = $verify.Open($outKorean, "HWPX", "")
  if (-not $ok) { throw "저장된 수정본을 한글로 다시 열지 못했습니다." }
  Write-Output "OK"
  Write-Output "ORIG=$($orig.FullName)"
  Write-Output "OUT=$outKorean"
  Write-Output "SIZE=$((Get-Item -LiteralPath $outKorean).Length)"
}
finally {
  try { $verify.Quit() } catch {}
}
