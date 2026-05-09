from pathlib import Path
import json
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path.cwd()
DL = Path.home() / "Downloads"
src = DL / "EmoNet_최종본.hwpx"
if not src.exists():
    src = max(
        [p for p in DL.glob("EmoNet*.hwpx") if p.stat().st_size > 2800000 and "수정본" not in p.name and "revision" not in p.name],
        key=lambda p: p.stat().st_size,
    )

with zipfile.ZipFile(src) as z:
    root = ET.fromstring(z.read("Contents/section0.xml"))

HP = "{http://www.hancom.co.kr/hwpml/2011/paragraph}"
paragraphs = []
for para in root.iter(HP + "p"):
    text = "".join(t.text or "" for t in para.iter(HP + "t")).strip()
    if text:
        paragraphs.append(text)

def starts(prefix: str) -> str:
    for text in paragraphs:
        if text.startswith(prefix):
            return text
    raise RuntimeError(f"paragraph not found: {prefix}")

terms = """2. 이론적 배경
가. 주요 용어의 정의
[수정] LLM(Large Language Model)은 대규모 언어 모델을 뜻한다. 본 탐구에서 LLM은 사용자의 감정을 직접 판정하는 최종 분류기가 아니라, 입력 발화와 프롬프트 조건을 바탕으로 응답을 생성하는 언어적 실행 장치로 사용된다.
[수정] EmoNet은 입력 발화에서 유발된 감정적 자극을 내부 node와 branch의 활성 흐름으로 표현하고, 그 trace를 응답 생성에 연결하는 계산적 감정 동역학 모델이다. 여기서 감정은 인간의 주관적 체험 그 자체가 아니라 응답을 조절하는 내부 상태 변수로 정의된다.
[수정] 내부 동역학은 한 번의 라벨 출력이 아니라 시간 tick에 따라 node activation, branch weight, inhibitory 또는 modulatory 효과가 변화하는 과정이다. 감정 동역학은 이 변화가 valence, arousal, action tendency와 연결되는 방식을 뜻한다.
[수정] affective stimulus는 감정 반응을 유발하는 입력 단서이다. node는 이러한 자극에 반응하는 계산 단위이며, excitatory node는 활성을 높이고, inhibitory node는 활성을 억제하며, modulatory node는 다른 node나 branch의 반응 방식을 조절한다.
[수정] branch는 내부 감정 흐름의 한 갈래이며, trace는 시간 순서대로 기록된 node와 branch의 활성 기록이다. trajectory는 trace가 시간에 따라 이동하는 경로이고, episode는 하나의 입력에 대해 형성된 비교적 완결된 감정 동역학 단위이다.
[수정] dominant branch는 특정 episode에서 가장 강하게 유지된 branch를 뜻한다. branch collapse는 다양한 branch가 살아 있어야 할 상황에서도 cooperativeness나 softness 쪽으로 과도하게 몰리는 현상이다. non-trivial affective trace는 단일 라벨이나 상투적 친절함으로 환원되지 않는 시간적 변화와 분기 구조를 가진 trace를 뜻한다.
[수정] calibration은 내부 파라미터를 조정하여 branch collapse나 style bias를 줄이는 과정이다. 파라미터는 node 가중치, branch threshold, decay, coupling strength처럼 모델의 동작을 정하는 수치이고, configuration은 이러한 파라미터 묶음과 실행 조건 전체를 뜻한다.
[수정] SOTA(State of the Art)는 해당 시점에서 가장 높은 성능을 보이는 최신 방법을 뜻한다. 누적 K는 상위 K개 활성 node를 누적하여 보는 분석 방식이다. raw는 보정 이전의 원자료 값을 뜻하고, bucket은 연속적인 값을 해석하기 쉬운 구간으로 묶은 범주이다. full58 실험은 58개 조건 또는 항목 전체를 포함한 비교 실험을 뜻한다. 신뢰 구간은 반복 측정에서 추정값이 어느 범위에 있을 가능성이 높은지를 보여 준다.
3. 선행 탐구"""

experiment = """5. 실험 환경과 전체 실험 절차
[수정] 본 탐구의 실험은 한국어 감성 대화 데이터와 EmoNet 내부 trace 산출물을 함께 사용하였다. 입력 데이터는 사용자의 발화, 상황 설명, 감정 라벨 또는 감정적 단서가 포함된 샘플로 구성되며, EmoNet은 이를 affective stimulus로 받아 내부 node activity와 branch trace를 생성하였다.
[수정] 전체 절차는 네 단계이다. 첫째, 입력 발화를 정리하고 동일한 응답 생성 프롬프트 형식으로 변환하였다. 둘째, EmoNet의 내부 node와 branch가 시간 tick에 따라 어떻게 활성화되는지 기록하였다. 셋째, trace를 episode 단위로 해석하여 valence, arousal, dominant branch, confidence를 산출하였다. 넷째, 생성 응답을 평가 지표에 따라 비교하고 style bias와 branch collapse 여부를 확인하였다.
[수정] 데이터셋의 한계도 함께 고려하였다. 일부 샘플은 cooperativeness나 negative-high arousal 영역에 자연스럽게 많이 분포할 수 있으므로, style target bias가 모델 문제인지 데이터셋 구성 문제인지 분리해서 해석해야 한다.
6. EmoNet 내부 trace 형성 실험"""

effects = """10. 기대효과
[수정] 첫째, EmoNet은 LLM 응답 생성에서 감정을 단순 라벨이나 친절한 문체로 처리하는 한계를 줄일 수 있다. 내부 trace를 사용하면 사용자의 발화를 바로 감정명으로 환원하지 않고, 시간적 변화와 행동 경향을 함께 반영할 수 있다.
[수정] 둘째, trace-sensitive 평가는 생성 응답의 품질을 더 세밀하게 비교할 수 있게 한다. 응답이 공손한지뿐 아니라, 내부 episode와 말투, 제안 행동, 안전성 판단이 서로 맞는지 확인할 수 있다.
[수정] 셋째, EmoNet의 trace와 branch 분석은 향후 개인화 대화 시스템, 상담 보조 시스템, 감정 변화 모니터링 시스템에서 해석 가능한 중간 근거로 활용될 수 있다.
11. 결론"""

replacements = []
for value in [
    "초록",
    starts("본 연구는 사용자의 감정을 분류하는 모델이 아니라"),
    starts("이 연구에서 감정은 인간과 같은 철학적인 감정을 뜻하지 않는다"),
    starts("실험은 세 단계로 진행하였다"),
    starts("이 결과만으로 EmoNet이 모든 대화 상황에서 기존 방식보다 낫다고"),
    starts("Keyword:"),
]:
    replacements.append({"find": value, "replace": ""})

replacements.extend([
    {"find": "2. 선행 연구 (정확한 출처와 함꼐 다시 쓰기)", "replace": terms},
    {"find": "3. EmoNet 구조", "replace": "4. EmoNet 구조"},
    {"find": "4. EmoNet 내부 trace 형성 실험", "replace": experiment},
    {"find": "5. trace의 emotion episode 해석", "replace": "7. trace의 emotion episode 해석"},
    {"find": "6. 응답 생성 및 평가", "replace": "8. 응답 생성 및 평가"},
    {"find": "7. 논의", "replace": "9. 고찰"},
    {"find": "8. 결론", "replace": effects},
    {"find": "1.1 문제 제기", "replace": "가. 문제 제기"},
    {"find": "1.2 연구 질문", "replace": "나. 탐구 질문"},
    {"find": "2.1 감정 인식과 감정 라벨링", "replace": "가. 감정 인식과 감정 라벨링"},
    {"find": "2.2 감정 응답 생성과 공감 대화", "replace": "나. 감정 응답 생성과 공감 대화"},
    {"find": "2.3 프롬프트 기반 감정 제어와 LLM judge", "replace": "다. 프롬프트 기반 감정 제어와 LLM judge"},
    {"find": "2.4 기존 접근과 EmoNet의 차이", "replace": "라. 기존 접근과 EmoNet의 차이"},
    {"find": "3.1 전체 구조", "replace": "가. 전체 구조"},
    {"find": "3.2 자극의 정의", "replace": "나. 자극의 정의"},
    {"find": "3.3 형식적 정의", "replace": "다. 형식적 정의"},
    {"find": "3.4 trace, branch, dominant branch", "replace": "라. trace, branch, dominant branch"},
    {"find": "3.5 emotion episode 해석", "replace": "마. emotion episode 해석"},
    {"find": "4.1 왜 이 실험이 필요한가", "replace": "가. 왜 이 실험이 필요한가"},
    {"find": "4.2 branch collapse와 보정", "replace": "나. branch collapse와 보정"},
    {"find": "4.3 branch collapse를 해결할 EmoNet 파라미터 calibration", "replace": "다. branch collapse를 해결할 EmoNet 파라미터 calibration"},
    {"find": "4.4 s_000555 실제 trace inspection", "replace": "라. s_000555 실제 trace inspection"},
    {"find": "4.5 기능적 노드 그룹과 대표 활성 경로", "replace": "마. 기능적 노드 그룹과 대표 활성 경로"},
    {"find": "4.6 주요 활성 node 분석", "replace": "바. 주요 활성 node 분석"},
    {"find": "4.7 남은 문제 -> 금방 해결 가능할것 같습니다. 일단 GPT로 땜빵 해뒀어요..", "replace": "사. 남은 문제"},
    {"find": "그림 9. trajectory-to-episode interpretation 결과 분포", "replace": "그림 9. trajectory-to-episode interpretation 결과 heatmap"},
    {"find": "넣을까요.....?", "replace": "[수정] 다음은 본 탐구에서 사용하는 episode 표현의 개념적 형식이다. 하나의 입력 s에 대해 EmoNet은 시간 tick t마다 node activation과 branch score를 기록하고, 이 기록 전체를 trace로 둔다. episode는 trace를 valence, arousal, dominant branch, confidence, evidence span으로 요약한 단위이다."},
])

structure = starts("EmoNet의 처리 흐름은 입력 발화")
replacements.append({
    "find": structure,
    "replace": structure.replace("응답 조건화", "응답 생성 프롬프트")
    + "\n[수정] 전체 프레임워크는 입력 발화, affective stimulus 추출, 내부 node-branch 동역학, trace-to-episode 해석, 응답 생성 프롬프트 연결의 네 층으로 볼 수 있다."
    + "\n[수정] 신경망 관점에서 EmoNet은 완전한 생물학적 뇌 모델이 아니라 기능적 node graph에 가깝다. 각 node는 특정 감정 단서나 행동 경향에 민감하게 반응하며, branch는 여러 node의 활성 조합이 특정 방향으로 모일 때 형성되는 경로이다.",
})

eval_text = starts("trace-sensitive 평가는 생성 응답이 trace로부터")
replacements.append({
    "find": eval_text,
    "replace": eval_text + "\n[수정] 평가 지표는 EmoNet의 목적에 맞추어 선정하였다. 일반적인 공감성 점수만 사용하면 모델이 항상 부드럽고 협조적인 말투로 수렴하는 style bias를 놓칠 수 있다. 따라서 trace_alignment, action_tendency_fit, style_target_match, safety_consistency를 함께 보았다.",
})

replacements.extend([
    {"find": "연구", "replace": "탐구"},
    {"find": "논의", "replace": "고찰"},
    {"find": "함꼐", "replace": "함께"},
    {"find": "할 수 는", "replace": "할 수는"},
    {"find": "형성 되", "replace": "형성되"},
    {"find": "종료 되", "replace": "종료되"},
])

out = ROOT / "tmp" / "hwp_replacements.json"
out.write_text(json.dumps(replacements, ensure_ascii=False, indent=2), encoding="utf-8")
print(src)
print(out)
print(len(replacements))
