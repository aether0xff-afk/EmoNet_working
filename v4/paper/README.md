# Paper Workspace

이 폴더는 `v4` 기준 논문 작성용 작업 디렉터리다.

## 현재 기준 원고

- Markdown canonical draft: `PAPER.md`
- Optional LaTeX scaffold: `main.tex`

지금 단계에서는 Markdown이 주 편집 포맷이고, LaTeX는 나중에 투고용으로 옮기기 위한 scaffold로 둔다.

## 현재 반영된 내용

- `PAPER_DRAFT_ko.md`의 핵심 서사
- `PAPER_ROADMAP_2026-04-10.md`의 주장/비주장 기준
- `paper_refresh_summary.json`의 branch/predictor/style bias 수치
- `paper_matrix_current_episode_v2_scored_summary.json`의 최신 generation 비교 수치
- 기존 refresh bundle figure 링크

## LaTeX 빌드

한국어 본문이 포함되어 있으므로 `xelatex` 기준으로 빌드한다.

```powershell
powershell -ExecutionPolicy Bypass -File .\build.ps1
```

빌드 산출물은 `build/`에 생성된다.

## 현재 비어 있는 부분

- 관련 연구 인용과 참고문헌
- 정식 저자 정보
- latest `episode_v2` generation figure 재생성
- qualitative case table

## 주의

현재 원고는 working draft다. generation 결과는 `episode_v2` summary를 기준으로 정리했고, refresh bundle 안의 baseline-only 표와 수치가 다르므로 최종 본문 정리 전에 source를 하나로 통일해야 한다.
