from pathlib import Path
import copy
import csv
import re
import shutil
import zipfile
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


ROOT = Path.cwd()
DL = Path.home() / "Downloads"
src = max(
    [p for p in DL.glob("EmoNet_최종본*.hwpx") if "수정본" not in p.name],
    key=lambda p: p.stat().st_size,
)
out = DL / "EmoNet_revision.hwpx"
work = ROOT / "tmp" / "hwpx_revision_final"
if work.exists():
    shutil.rmtree(work)
work.mkdir(parents=True)
with zipfile.ZipFile(src) as z:
    z.extractall(work)


def make_heatmap(path: Path) -> None:
    counts = {
        ("negative", "high"): 85,
        ("negative", "medium"): 3,
        ("negative", "low"): 1,
        ("mixed", "high"): 17,
        ("mixed", "medium"): 1,
        ("mixed", "low"): 1,
        ("positive", "high"): 7,
        ("positive", "medium"): 5,
        ("positive", "low"): 0,
    }
    csv_path = ROOT / "v4" / "outputs" / "research" / "trajectory_batch_matrix120_v1_gpt54" / "episode_summary.csv"
    if csv_path.exists():
        counts = {}
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                key = ((row.get("valence") or "").strip(), (row.get("arousal") or "").strip())
                if key[0] and key[1]:
                    counts[key] = counts.get(key, 0) + 1

    valences = ["negative", "mixed", "positive"]
    arousals = ["high", "medium", "low"]
    data = [[counts.get((v, a), 0) for a in arousals] for v in valences]
    maxv = max(max(r) for r in data) or 1

    width, height = 2350, 994
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    def font(size: int, bold: bool = False):
        names = ["arialbd.ttf", "calibrib.ttf"] if bold else ["arial.ttf", "calibri.ttf"]
        for name in names:
            candidate = Path("C:/Windows/Fonts") / name
            if candidate.exists():
                return ImageFont.truetype(str(candidate), size)
        return ImageFont.load_default()

    def color(value: int):
        t = value / maxv
        stops = [(255, 242, 184), (252, 140, 61), (179, 0, 38)]
        if t < 0.5:
            a, b, tt = stops[0], stops[1], t / 0.5
        else:
            a, b, tt = stops[1], stops[2], (t - 0.5) / 0.5
        return tuple(int(a[i] * (1 - tt) + b[i] * tt) for i in range(3))

    title_font = font(58, True)
    label_font = font(42)
    cell_font = font(62, True)
    axis_font = font(38)
    draw.text((width // 2, 70), "trajectory-to-episode interpretation heatmap", anchor="mm", fill="#222222", font=title_font)
    left, top, cell_w, cell_h = 440, 210, 445, 190
    for j, arousal in enumerate(arousals):
        draw.text((left + j * cell_w + cell_w / 2, top - 55), arousal, anchor="mm", fill="#222222", font=label_font)
    for i, valence in enumerate(valences):
        draw.text((left - 55, top + i * cell_h + cell_h / 2), valence, anchor="rm", fill="#222222", font=label_font)
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            x0, y0 = left + j * cell_w, top + i * cell_h
            x1, y1 = x0 + cell_w, y0 + cell_h
            draw.rectangle([x0, y0, x1, y1], fill=color(value), outline="white", width=6)
            draw.text(
                ((x0 + x1) / 2, (y0 + y1) / 2),
                str(value),
                anchor="mm",
                fill=("white" if value >= maxv * 0.55 else "#222222"),
                font=cell_font,
            )
    draw.rectangle([left, top, left + cell_w * 3, top + cell_h * 3], outline="#333333", width=3)
    draw.text((left + cell_w * 1.5, height - 120), "arousal", anchor="mm", fill="#222222", font=axis_font)
    draw.text((130, top + cell_h * 1.5), "valence", anchor="mm", fill="#222222", font=axis_font)
    bar_x, bar_y, bar_w, bar_h = 1900, 230, 58, 510
    for y in range(bar_h):
        draw.line([(bar_x, bar_y + y), (bar_x + bar_w, bar_y + y)], fill=color(maxv * (1 - y / (bar_h - 1))))
    draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], outline="#333333", width=2)
    draw.text((bar_x + 90, bar_y), str(maxv), anchor="lm", font=axis_font, fill="#222222")
    draw.text((bar_x + 90, bar_y + bar_h), "0", anchor="lm", font=axis_font, fill="#222222")
    draw.text((bar_x + bar_w / 2, bar_y + bar_h + 60), "count", anchor="mm", font=axis_font, fill="#222222")
    img.save(path)


make_heatmap(work / "BinData" / "image9.PNG")

# Replacing an embedded image invalidates Hancom's internal hashkey metadata.
# The hashkey is optional in HWPX package manifests, so remove it to avoid
# triggering Hancom's tamper/damage warning on the edited package.
hpf_path = work / "Contents" / "content.hpf"
hpf_text = hpf_path.read_text(encoding="utf-8")
hpf_text = re.sub(r'\s+hashkey="[^"]*"', "", hpf_text)
hpf_path.write_text(hpf_text, encoding="utf-8")

for prefix, uri in [
    ("hp", "http://www.hancom.co.kr/hwpml/2011/paragraph"),
    ("hc", "http://www.hancom.co.kr/hwpml/2011/core"),
    ("ha", "http://www.hancom.co.kr/hwpml/2011/app"),
    ("hp10", "http://www.hancom.co.kr/hwpml/2016/paragraph"),
    ("hs", "http://www.hancom.co.kr/hwpml/2011/section"),
    ("hhs", "http://www.hancom.co.kr/hwpml/2011/history"),
    ("hm", "http://www.hancom.co.kr/hwpml/2011/master-page"),
    ("hpf", "http://www.hancom.co.kr/schema/2011/hpf"),
    ("dc", "http://purl.org/dc/elements/1.1/"),
    ("opf", "http://www.idpf.org/2007/opf/"),
    ("ooxmlchart", "http://www.hancom.co.kr/hwpml/2016/ooxmlchart"),
    ("hwpunitchar", "http://www.hancom.co.kr/hwpml/2016/HwpUnitChar"),
    ("epub", "http://www.idpf.org/2007/ops"),
    ("config", "urn:oasis:names:tc:opendocument:xmlns:config:1.0"),
]:
    ET.register_namespace(prefix, uri)

sec = work / "Contents" / "section0.xml"
tree = ET.parse(sec)
root = tree.getroot()
HP = "{http://www.hancom.co.kr/hwpml/2011/paragraph}"


def paras():
    return list(root.iter(HP + "p"))


def txt(p):
    return "".join(t.text or "" for t in p.iter(HP + "t"))


def settxt(p, value: str):
    ts = list(p.iter(HP + "t"))
    if not ts:
        return
    ts[0].text = value
    for t in ts[1:]:
        t.text = ""


def pmap():
    return {c: p for p in root.iter() for c in p}


def starts(prefix: str):
    for p in paras():
        if txt(p).strip().startswith(prefix):
            return p
    return None


def exact(value: str):
    for p in paras():
        if txt(p).strip() == value:
            return p
    return None


def newlike(template, value: str):
    global next_para_id
    q = copy.deepcopy(template)
    q.set("id", str(next_para_id))
    next_para_id += 1
    settxt(q, value)
    return q


def before(ref, q):
    parents = pmap()
    parent = parents[ref]
    parent.insert(list(parent).index(ref), q)


def after(ref, q):
    parents = pmap()
    parent = parents[ref]
    parent.insert(list(parent).index(ref) + 1, q)


def remove(p):
    parents = pmap()
    parent = parents.get(p)
    if parent is not None:
        parent.remove(p)


body_template = starts("현재의 LLM은") or paras()[0]
heading_template = exact("1. 서론") or body_template
next_para_id = max(int(p.get("id", "0")) for p in paras() if p.get("id", "0").isdigit()) + 1

for t in root.iter(HP + "t"):
    if t.text:
        t.text = t.text.replace("연구", "탐구").replace("논의", "고찰").replace("함꼐", "함께")

for p in list(paras()):
    value = txt(p).strip()
    if any(
        value.startswith(prefix)
        for prefix in [
            "초록",
            "본 탐구는 사용자의 감정을 분류하는 모델이 아니라",
            "이 탐구에서 감정은 인간과 같은 철학적인 감정을 뜻하지 않는다",
            "실험은 세 단계로 진행하였다",
            "이 결과만으로 EmoNet이 모든 대화 상황에서 기존 방식보다 낫다고 할 수 는 없다",
            "Keyword:",
        ]
    ):
        remove(p)

p = exact("이은세")
if p is not None:
    after(
        p,
        newlike(
            body_template,
            "[수정 표시 안내] 본 수정본에서 새로 추가하거나 크게 고친 문단은 [수정]으로 표시하였다. 기존 본문은 가능한 한 유지하되, 요약문 삭제, 용어 정리 보강, 절 구성 재배치, 그림 9 heatmap 교체, 평가 지표 선정 이유와 기대효과 추가를 반영하였다.",
        ),
    )

for old, new in [
    ("2. 선행 탐구 (정확한 출처와 함께 다시 쓰기)", "3. 선행 탐구"),
    ("3. EmoNet 구조", "4. EmoNet 구조"),
    ("4. EmoNet 내부 trace 형성 실험", "6. EmoNet 내부 trace 형성 실험"),
    ("5. trace의 emotion episode 해석", "7. trace의 emotion episode 해석"),
    ("6. 응답 생성 및 평가", "8. 응답 생성 및 평가"),
    ("7. 고찰", "9. 고찰"),
    ("8. 결론", "11. 결론"),
]:
    p = exact(old)
    if p is not None:
        settxt(p, new)

ref = exact("3. 선행 탐구")
if ref is not None:
    terms = [
        "2. 이론적 배경",
        "가. 주요 용어의 정의",
        "[수정] LLM(Large Language Model)은 대규모 언어 모델을 뜻한다. 본 탐구에서 LLM은 사용자의 감정을 직접 판정하는 최종 분류기가 아니라, 입력 발화와 프롬프트 조건을 바탕으로 응답을 생성하는 언어적 실행 장치로 사용된다.",
        "[수정] EmoNet은 입력 발화에서 유발된 감정적 자극을 내부 node와 branch의 활성 흐름으로 표현하고, 그 trace를 응답 생성에 연결하는 계산적 감정 동역학 모델이다. 여기서 감정은 인간의 주관적 체험 그 자체가 아니라 응답을 조절하는 내부 상태 변수로 정의된다.",
        "[수정] 내부 동역학은 한 번의 라벨 출력이 아니라 시간 tick에 따라 node activation, branch weight, inhibitory 또는 modulatory 효과가 변화하는 과정이다. 감정 동역학은 이 변화가 valence, arousal, action tendency와 연결되는 방식을 뜻한다.",
        "[수정] affective stimulus는 감정 반응을 유발하는 입력 단서이다. 본 탐구에서는 사용자의 발화, 상황 정보, 표현 강도, 위험 신호가 모두 자극으로 작동할 수 있다. node는 이러한 자극에 반응하는 계산 단위이며, excitatory node는 활성을 높이고, inhibitory node는 활성을 억제하며, modulatory node는 다른 node나 branch의 반응 방식을 조절한다.",
        "[수정] 응답 생성의 프롬프트는 기존의 응답 조건화라는 표현을 더 명확히 바꾼 용어이다. 이는 LLM에게 단순히 친절하게 답하라고 지시하는 것이 아니라, EmoNet trace에서 나온 내부 상태와 행동 경향을 응답 말투와 내용 선택에 반영하도록 연결하는 입력 형식이다.",
        "[수정] branch는 내부 감정 흐름의 한 갈래이며, trace는 시간 순서대로 기록된 node와 branch의 활성 기록이다. trajectory는 trace가 시간에 따라 이동하는 경로이고, episode는 하나의 입력에 대해 형성된 비교적 완결된 감정 동역학 단위이다. episode의 주요 필드는 valence, arousal, dominant branch, confidence, evidence trace로 해석된다.",
        "[수정] dominant branch는 특정 episode에서 가장 강하게 유지된 branch를 뜻한다. branch collapse는 다양한 branch가 살아 있어야 할 상황에서도 특정 branch, 예를 들어 cooperativeness나 softness 쪽으로 과도하게 몰리는 현상이다. non-trivial affective trace는 단일 라벨이나 상투적 친절함으로 환원되지 않는, 시간적 변화와 분기 구조를 가진 trace를 뜻한다.",
        "[수정] calibration은 내부 파라미터를 조정하여 branch collapse나 style bias를 줄이는 과정이다. 파라미터는 node 가중치, branch threshold, decay, coupling strength처럼 모델의 동작을 정하는 수치이고, configuration은 이러한 파라미터 묶음과 실행 조건 전체를 뜻한다.",
        "[수정] SOTA(State of the Art)는 해당 시점에서 가장 높은 성능을 보이는 최신 방법을 뜻한다. 본 탐구는 SOTA 감정 분류 모델을 대체하려는 것이 아니라, LLM 응답 생성에서 내부 감정 trace를 활용할 수 있는지를 탐구한다.",
        "[수정] 누적 K는 상위 K개 활성 node를 누적하여 보는 분석 방식이다. raw는 보정 이전의 원자료 값을 뜻하고, bucket은 연속적인 값을 해석하기 쉬운 구간으로 묶은 범주이다. full58 실험은 58개 조건 또는 항목 전체를 포함한 비교 실험을 뜻한다. 신뢰 구간은 반복 측정에서 추정값이 어느 범위에 있을 가능성이 높은지를 보여 주며, 겹침 여부만으로 효과의 유무를 단정하지 않고 평균 차이와 표본 수를 함께 해석해야 한다.",
    ]
    for value in terms:
        before(ref, newlike(heading_template if re.match(r"^\d+\.", value) else body_template, value))

p = starts("EmoNet의 처리 흐름은 입력 발화")
if p is not None:
    last = p
    for value in [
        "[수정] 전체 프레임워크는 입력 발화, affective stimulus 추출, 내부 node-branch 동역학, trace-to-episode 해석, 응답 생성 프롬프트 연결의 네 층으로 볼 수 있다. 입력 발화가 곧바로 감정 라벨로 변환되는 것이 아니라, 여러 node의 활성 변화와 branch 경쟁을 거친 뒤 episode로 요약된다.",
        "[수정] 신경망 관점에서 EmoNet은 완전한 생물학적 뇌 모델이 아니라 기능적 node graph에 가깝다. 각 node는 특정 감정 단서나 행동 경향에 민감하게 반응하며, branch는 여러 node의 활성 조합이 특정 방향으로 모일 때 형성되는 경로이다. 따라서 EmoNet의 핵심은 단일 출력층보다 중간 활성 trace를 해석 가능하게 남긴다는 점이다.",
        "[수정] 그림 1은 이 구조를 요약한다. 사용자의 발화는 affective stimulus로 정규화되고, 내부 node graph에서 excitatory, inhibitory, modulatory 효과를 거친다. 이후 dominant branch와 episode 필드가 산출되며, 이 값이 응답 생성 프롬프트에 들어가 LLM의 말투와 내용 선택을 조절한다.",
    ]:
        q = newlike(body_template, value)
        after(last, q)
        last = q

p = exact("넣을까요.....?")
if p is not None:
    settxt(p, "[수정] 다음은 본 탐구에서 사용하는 episode 표현의 개념적 형식이다. 하나의 입력 s에 대해 EmoNet은 시간 tick t마다 node activation과 branch score를 기록하고, 이 기록 전체를 trace로 둔다. episode는 trace를 valence, arousal, dominant branch, confidence, evidence span으로 요약한 단위이다.")

ref = exact("6. EmoNet 내부 trace 형성 실험")
if ref is not None:
    section = [
        "5. 실험 환경과 전체 실험 절차",
        "[수정] 본 탐구의 실험은 한국어 감성 대화 데이터와 EmoNet 내부 trace 산출물을 함께 사용하였다. 입력 데이터는 사용자의 발화, 상황 설명, 감정 라벨 또는 감정적 단서가 포함된 샘플로 구성되며, EmoNet은 이를 affective stimulus로 받아 내부 node activity와 branch trace를 생성하였다.",
        "[수정] 전체 절차는 네 단계이다. 첫째, 입력 발화를 정리하고 동일한 응답 생성 프롬프트 형식으로 변환하였다. 둘째, EmoNet의 내부 node와 branch가 시간 tick에 따라 어떻게 활성화되는지 기록하였다. 셋째, trace를 episode 단위로 해석하여 valence, arousal, dominant branch, confidence를 산출하였다. 넷째, 생성 응답을 평가 지표에 따라 비교하고 style bias와 branch collapse 여부를 확인하였다.",
        "[수정] 실험 환경에서는 동일한 입력에 대해 조건만 다르게 하여 raw trace, calibrated trace, 응답 생성 결과를 비교하였다. 이를 통해 단순히 응답 문장이 부드러워졌는지가 아니라, 내부 감정 동역학이 응답의 말투와 행동 경향에 실제로 반영되었는지를 확인하였다.",
        "[수정] 데이터셋의 한계도 함께 고려하였다. 일부 샘플은 cooperativeness나 negative-high arousal 영역에 자연스럽게 많이 분포할 수 있으므로, style target bias가 모델 문제인지 데이터셋 구성 문제인지 분리해서 해석해야 한다. 따라서 본 탐구에서는 단일 평균값보다 trace 분포, episode heatmap, 사례 분석을 함께 사용하였다.",
    ]
    for value in section:
        before(ref, newlike(heading_template if re.match(r"^\d+\.", value) else body_template, value))

for prefix, body_text, short in [
    ("그림 5. s_000555 사례의 내부 trace activity", "[수정] 그림 5는 s_000555 사례에서 시간 tick에 따라 내부 node가 어떻게 활성화되는지를 보여 준다. 음영은 dormant threshold 아래에 있는 node를 뜻하며, 실선은 특정 tick에서 실제로 살아 있는 node activity를 뜻한다.", "그림 5. s_000555 내부 trace activity"),
    ("그림 6. s_000555 사례의 raw affect signal trajectory", "[수정] 그림 6은 같은 사례의 raw affect signal trajectory를 나타낸다. 이 그래프는 보정 이전의 감정 신호가 시간에 따라 어느 방향으로 이동하는지 보여 주며, stimulus의 표면 라벨과 내부 episode가 항상 같은 방향으로 움직이지 않을 수 있음을 확인하게 해 준다.", "그림 6. s_000555 raw affect signal trajectory"),
    ("그림 7. tick별 기능적 노드 그룹 활성도 heatmap.", "[수정] 그림 7은 tick별 기능적 node 그룹 활성도를 heatmap으로 정리한 것이다. 이를 통해 단일 node보다 기능적 그룹 단위에서 어떤 반응 경로가 강해졌는지 비교할 수 있다.", "그림 7. tick별 기능적 노드 그룹 활성도 heatmap"),
    ("그림 8. s_000555 주요 활성 node.", "[수정] 그림 8은 s_000555에서 주요하게 활성화된 node를 상위 K 방식으로 보여 준다. 누적 K 분석은 특정 시점의 최대값뿐 아니라 episode 전체에서 반복적으로 기여한 node를 확인하기 위해 사용하였다.", "그림 8. s_000555 주요 활성 node"),
]:
    p = starts(prefix)
    if p is not None:
        before(p, newlike(body_template, body_text))
        settxt(p, short)

p = starts("trace-to-episode 해석은 120개 샘플에서 수행")
if p is not None:
    settxt(p, "trace-to-episode 해석은 120개 샘플에서 수행되었다. 평균 confidence는 0.9293이었다. valence 분포는 negative 89개, mixed 19개, positive 12개였고, arousal 분포는 high 109개, medium 9개, low 2개였다. 그림 9는 이 분포를 valence와 arousal의 교차 heatmap으로 나타낸 것이다. 이를 통해 대부분의 trace가 high arousal-negative 영역에 집중되어 있음을 확인할 수 있다.")
p = starts("그림 9. trajectory-to-episode interpretation 결과")
if p is not None:
    settxt(p, "그림 9. trajectory-to-episode interpretation 결과 heatmap")

p = starts("s_000555는 stimulus 축과 emotion episode의 차이")
if p is not None:
    last = p
    for value in [
        "[수정] s_003491은 표면적으로는 단순한 부정 정서처럼 보이지만, trace에서는 회피와 자기보호 계열 branch가 함께 활성화된 사례이다. 이 경우 응답은 단순한 위로나 긍정 유도보다 사용자가 안전하게 상황을 정리하도록 돕는 방향이 더 적절하다.",
        "[수정] s_000149는 감정 강도가 낮아 보이는 입력에서도 특정 node가 반복적으로 활성화되면 episode confidence가 높아질 수 있음을 보여 준다. 이 사례는 EmoNet이 문장 표면의 강한 감정어보다 trace의 누적 패턴을 중시한다는 점을 설명한다.",
    ]:
        q = newlike(body_template, value)
        after(last, q)
        last = q

p = starts("trace-sensitive 평가는 생성 응답이 trace로부터")
if p is not None:
    last = p
    for value in [
        "[수정] 평가 지표는 EmoNet의 목적에 맞추어 선정하였다. 일반적인 공감성 점수만 사용하면 모델이 항상 부드럽고 협조적인 말투로 수렴하는 style bias를 놓칠 수 있다. 따라서 본 탐구에서는 trace_alignment, action_tendency_fit, style_target_match, safety_consistency를 함께 보았다.",
        "[수정] action_tendency_fit은 episode가 요구하는 행동 경향과 응답이 실제로 제안하는 행동이 맞는지를 확인하기 위한 지표이다. style_target_match는 응답의 말투가 내부 trace에서 나온 목표 style과 일치하는지를 보기 위한 지표이며, safety_consistency는 위험 신호가 있는 경우 응답이 부적절하게 낙관적이거나 회피적으로 흐르지 않는지를 확인하기 위해 사용하였다.",
    ]:
        q = newlike(body_template, value)
        after(last, q)
        last = q

p = exact("11. 결론")
if p is not None:
    effects = [
        "10. 기대효과",
        "[수정] 첫째, EmoNet은 LLM 응답 생성에서 감정을 단순 라벨이나 친절한 문체로 처리하는 한계를 줄일 수 있다. 내부 trace를 사용하면 사용자의 발화를 바로 감정명으로 환원하지 않고, 시간적 변화와 행동 경향을 함께 반영할 수 있다.",
        "[수정] 둘째, trace-sensitive 평가는 생성 응답의 품질을 더 세밀하게 비교할 수 있게 한다. 응답이 공손한지뿐 아니라, 내부 episode와 말투, 제안 행동, 안전성 판단이 서로 맞는지 확인할 수 있다.",
        "[수정] 셋째, EmoNet의 trace와 branch 분석은 향후 개인화 대화 시스템, 상담 보조 시스템, 감정 변화 모니터링 시스템에서 해석 가능한 중간 근거로 활용될 수 있다. 특히 branch collapse나 style bias를 진단할 수 있다는 점은 LLM 응답 제어의 실험적 근거가 된다.",
        "[수정] 다만 본 탐구의 기대효과는 모든 감정 대화 문제를 해결한다는 의미가 아니다. 데이터셋 구성, 평가자의 해석 기준, LLM 자체의 문체 편향이 결과에 영향을 줄 수 있으므로 후속 탐구에서는 더 큰 표본과 다양한 도메인에서 검증해야 한다.",
    ]
    for value in effects:
        before(p, newlike(heading_template if re.match(r"^\d+\.", value) else body_template, value))

for prefix, cite in [
    ("감정 인식 탐구는 텍스트", "[1][2][3]"),
    ("감정 라벨은 응답 생성을", "[4]"),
    ("Emotional Chatting Machine", "[5][6][7][8]"),
    ("LLM 기반 시스템에서는", "[9][10]"),
]:
    p = starts(prefix)
    if p is not None and cite not in txt(p):
        settxt(p, txt(p).rstrip().rstrip(".") + cite + ".")

for p in paras():
    value = txt(p).strip()
    if value.startswith("4.7 남은 문제") or value.startswith("6.7 남은 문제"):
        settxt(p, "사. 남은 문제")
    elif value.startswith("9.3 style bias와 surface softening") or value.startswith("7.3 style bias와 surface softening"):
        settxt(p, "다. style bias와 surface softening")
    elif value.startswith("9.4 한계") or value.startswith("7.4 한계"):
        settxt(p, "라. 한계")

letters = ["가", "나", "다", "라", "마", "바", "사", "아", "자", "차", "카", "타", "파", "하"]
for p in paras():
    value = txt(p).strip()
    m3 = re.match(r"^(\d+)\.(\d+)\.(\d+)\s+(.+)$", value)
    m2 = re.match(r"^(\d+)\.(\d+)\s+(.+)$", value)
    if m3:
        idx = int(m3.group(3)) - 1
        settxt(p, f"{letters[idx] if 0 <= idx < len(letters) else idx + 1}) {m3.group(4)}")
    elif m2:
        idx = int(m2.group(2)) - 1
        settxt(p, f"{letters[idx] if 0 <= idx < len(letters) else idx + 1}. {m2.group(3)}")

for p in paras():
    value = txt(p)
    new_value = value
    for bad in [
        " - 해결할거임",
        "금방 해결 가능할것 같습니다. 일단 GPT로 땜빵 해뒀어요..",
        "정확한 출처와 함께 다시 쓰기",
        "-> 나중에 모아볼게요.. 필요가 있을지는 모르겠지만..",
    ]:
        new_value = new_value.replace(bad, "")
    new_value = new_value.replace("할 수 는", "할 수는").replace("형성 되", "형성되").replace("종료 되", "종료되")
    if new_value != value:
        settxt(p, new_value)

tree.write(sec, encoding="UTF-8", xml_declaration=True)

if out.exists():
    out.unlink()

# HWPX/Hancom is sensitive to package layout. Preserve the original entry
# order, compression method, flags, timestamps, and attributes instead of
# rebuilding the archive as a generic zip.
with zipfile.ZipFile(src, "r") as original, zipfile.ZipFile(out, "w") as z:
    for info in original.infolist():
        replacement = work / info.filename
        data = replacement.read_bytes() if replacement.exists() else original.read(info.filename)
        new_info = zipfile.ZipInfo(info.filename, date_time=info.date_time)
        new_info.compress_type = info.compress_type
        new_info.comment = info.comment
        new_info.extra = info.extra
        new_info.internal_attr = info.internal_attr
        new_info.external_attr = info.external_attr
        new_info.create_system = info.create_system
        new_info.flag_bits = info.flag_bits
        z.writestr(new_info, data)

with zipfile.ZipFile(out) as z:
    text = z.read("Contents/section0.xml").decode("utf-8")
    checks = {
        "revision": "[수정]" in text,
        "abstract_removed": "초록" not in text,
        "fig9_heatmap": "그림 9. trajectory-to-episode interpretation 결과 heatmap" in text,
        "expected_effects": "10. 기대효과" in text,
        "discussion_renamed": "9. 고찰" in text,
        "research_word_removed": "연구" not in text,
        "discussion_word_removed": "논의" not in text,
    }
print(f"SRC={src}")
print(f"OUT={out}")
print(f"SIZE={out.stat().st_size}")
print(checks)
