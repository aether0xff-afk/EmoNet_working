$ErrorActionPreference = "Stop"

$downloads = Join-Path $env:USERPROFILE "Downloads"
$orig = Join-Path $downloads "EmoNet_최종본 backup.hwpx"
if (-not (Test-Path $orig)) {
  $orig = Join-Path $downloads "EmoNet_최종본 - 복사본.hwpx"
}
if (-not (Test-Path $orig)) {
  throw "보존용 원본을 찾지 못했습니다."
}

$outAscii = Join-Path $downloads "EmoNet_preserve_edit.hwpx"
$outKorean = Join-Path $downloads "EmoNet_최종본_수정본.hwpx"
Remove-Item -LiteralPath $outAscii -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $outKorean -Force -ErrorAction SilentlyContinue

$hwp = New-Object -ComObject HWPFrame.HwpObject

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
  $hwp.HAction.Execute("AllReplace", $set.HSet) | Out-Null
}

function Insert-ChunkedText {
  param([Parameter(Mandatory=$true)][string]$Text)
  for ($i = 0; $i -lt $Text.Length; $i += 1200) {
    $len = [Math]::Min(1200, $Text.Length - $i)
    $hwp.Insert($Text.Substring($i, $len), 0, 0)
    Start-Sleep -Milliseconds 80
  }
}

try {
  $hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule") | Out-Null
  try { $hwp.XHwpWindows.Item(0).Visible = $false } catch {}
  $opened = $hwp.Open($orig, "HWPX", "forceopen:true")
  if (-not $opened) { throw "한글 COM으로 원본을 열지 못했습니다: $orig" }

  # Safe short replacements only. Tables, figures, and existing object layout remain intact.
  Replace-AllText "연구" "탐구"
  Replace-AllText "논의" "고찰"
  Replace-AllText "함꼐" "함께"
  Replace-AllText "1.1 문제 제기" "가. 문제 제기"
  Replace-AllText "1.2 탐구 질문" "나. 탐구 질문"
  Replace-AllText "1.2 연구 질문" "나. 탐구 질문"
  Replace-AllText "그림 9. trajectory-to-episode interpretation 결과 분포" "그림 9. trajectory-to-episode interpretation 결과 heatmap"
  Replace-AllText "넣을까요.....?" "[수정] episode는 하나의 입력에 대해 형성된 trace를 valence, arousal, dominant branch, confidence로 요약한 감정 동역학 단위이다."

  $append = @"


12. 수정 보강 내용
[수정] 이 보존형 수정본은 기존 표, 그림, 본문 배치, 캡션, 원본 개체를 삭제하지 않고 유지한 상태에서 보강 내용을 추가하였다. 기존 원고를 크게 훼손하지 않기 위해 표와 그림은 원래 위치에 남겨 두었다.
[수정] 용어 정리: LLM은 대규모 언어 모델이며, 본 탐구에서는 감정 판정기가 아니라 EmoNet trace를 언어 응답으로 실현하는 장치이다. EmoNet은 입력 발화가 에이전트 내부에 유발한 affective stimulus를 node, branch, trace, trajectory, episode로 기록하고 응답 생성 프롬프트에 연결하는 계산적 감정 동역학 모델이다.
[수정] branch는 내부 감정 흐름의 한 갈래이고, trace는 시간 순서대로 기록된 node와 branch의 활성 기록이다. trajectory는 trace가 시간에 따라 이동하는 경로이며, episode는 하나의 입력에서 형성된 비교적 완결된 감정 동역학 단위이다. dominant branch는 episode에서 가장 강하게 유지된 branch이고, branch collapse는 다양한 branch가 살아 있어야 할 상황에서 특정 branch로 과도하게 몰리는 현상이다.
[수정] calibration은 branch collapse와 style bias를 줄이기 위해 파라미터와 configuration을 조정하는 과정이다. excitatory node는 활성을 높이고, inhibitory node는 활성을 억제하며, modulatory node는 다른 node나 branch의 반응 방식을 조절한다. raw는 보정 전 원자료, bucket은 값을 해석 가능한 구간으로 묶은 범주, full58 실험은 58개 조건 전체를 포함한 비교 실험을 뜻한다.
[수정] 실험 절차는 입력 발화 정리, EmoNet trace 산출, trace-to-episode 해석, 응답 생성 및 평가의 순서로 진행된다. 데이터셋이 특정 정서나 cooperativeness 계열에 치우쳐 있을 수 있으므로, 평균값만 보지 않고 trace 분포, episode heatmap, 사례 분석을 함께 해석해야 한다.
[수정] 평가 지표는 일반적인 자연성만으로는 style bias를 놓칠 수 있기 때문에 trace_alignment, action_tendency_fit, style_target_match, safety_consistency를 함께 사용하였다.
[수정] 기대효과는 LLM 응답 생성에서 감정을 단순 라벨이나 친절한 문체로 처리하는 한계를 줄이고, 내부 trace와 응답 말투 사이의 연결을 해석 가능하게 만드는 데 있다.
[수정] 그림 9 heatmap은 아래에 추가하였다.
"@
  $hwp.Run("MoveDocEnd") | Out-Null
  Insert-ChunkedText $append

  $img = Join-Path (Get-Location) "tmp\hwpx_revision_final\BinData\image9.PNG"
  if (Test-Path $img) {
    $hwp.InsertPicture($img, $true, 0, $false, $false, 0, 0, 0) | Out-Null
  }

  $saved = $hwp.SaveAs($outAscii, "HWPX", "")
  if (-not $saved) { throw "SaveAs 실패" }
}
finally {
  try { $hwp.Quit() } catch {}
}

Copy-Item -LiteralPath $outAscii -Destination $outKorean -Force

Start-Sleep -Seconds 2
$verify = New-Object -ComObject HWPFrame.HwpObject
try {
  $ok = $verify.Open($outKorean, "HWPX", "")
  if (-not $ok) { throw "저장된 수정본을 다시 열지 못했습니다." }
  Write-Output "OK"
  Write-Output "ORIG=$orig"
  Write-Output "OUT=$outKorean"
  Write-Output "SIZE=$((Get-Item -LiteralPath $outKorean).Length)"
}
finally {
  try { $verify.Quit() } catch {}
}
