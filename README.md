# EmoNet Working Tree

이 저장소는 단일 패키지가 아니라 버전별 연구 작업선을 함께 보관하는 워크트리다.
현재 기준으로는 각 버전 디렉터리를 독립 작업 단위로 취급한다.

## Version Map

- `v1`
  - 초기 `emotion_z_pipeline` / GUI 실험선
  - 실행 의존성은 [v1/requirements.txt](C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v1/requirements.txt)
- `v2`
  - 모듈형 PyTorch MVP
  - 핵심 설명은 [v2/emonet/README.md](C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v2/emonet/README.md)
  - 실행 의존성은 [v2/requirements.txt](C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v2/requirements.txt)
- `v3`
  - self-contained legacy research line
  - branch calibration / trajectory / paper 정리본이 남아 있는 안정 스냅샷
  - 실행 의존성은 [v3/requirements.txt](C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v3/requirements.txt)
- `v4`
  - 현재 active 작업선
  - 개요는 [v4/README.md](C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v4/README.md)
  - 실행 의존성은 [v4/requirements.txt](C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v4/requirements.txt)
- `v5`
  - Luca형 캐릭터 대화 MVP 작업선
  - `v4` runtime을 기반으로 캐릭터 카드, 세션 기억, v3.1 trace-as-emotion 원칙을 적용
  - 개요는 [v5/README.md](C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v5/README.md)
  - 실행 의존성은 [v5/requirements.txt](C:/Users/esl01/OneDrive/문서/GitHub/EmoNet_working/v5/requirements.txt)

## Rule Of Thumb

- 루트에서는 버전 인덱싱과 공용 기록만 본다.
- 실제 실행, 테스트, 문서 확인은 해당 버전 디렉터리 안에서 수행한다.
- `v3`, `v4`는 루트의 `encoder-ML testing` 같은 공유 폴더를 기본 경로로 사용하지 않도록 정리했다.

## Quick Start

```powershell
cd .\v4
python -m emonet.cli --help
python -m unittest discover -s tests -v
streamlit run .\streamlit_app.py
```

```powershell
cd .\v5
python -m unittest discover -s tests -v
python .\local_gui.py
```
