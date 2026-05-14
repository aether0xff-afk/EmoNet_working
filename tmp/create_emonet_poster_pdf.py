from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import KeepInFrame, Paragraph, Spacer


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf"
OUT.mkdir(parents=True, exist_ok=True)
PDF_PATH = OUT / "1308_이은세_2026_1학기_ArtScience_창의융합_오디세이_포스터_EmoNet.pdf"

PAGE_W, PAGE_H = 1218, 1544
BLUE = colors.HexColor("#0718ff")
TEXT = colors.HexColor("#111111")
LIGHT_BLUE = colors.HexColor("#eff5ff")
LIGHT_GREEN = colors.HexColor("#f1fbf6")


def register_fonts() -> tuple[str, str]:
    regular = Path(r"C:\Windows\Fonts\NotoSansKR-VF.ttf")
    bold = Path(r"C:\Windows\Fonts\malgunbd.ttf")
    if regular.exists():
        pdfmetrics.registerFont(TTFont("KR", str(regular)))
    else:
        pdfmetrics.registerFont(TTFont("KR", r"C:\Windows\Fonts\malgun.ttf"))
    pdfmetrics.registerFont(TTFont("KRB", str(bold)))
    return "KR", "KRB"


FONT, FONT_B = register_fonts()


def para_style(size=18, leading=None, bold=False, align=TA_LEFT, color=TEXT):
    return ParagraphStyle(
        name=f"s{size}{'b' if bold else ''}{align}",
        fontName=FONT_B if bold else FONT,
        fontSize=size,
        leading=leading or size * 1.35,
        textColor=color,
        alignment=align,
        wordWrap="CJK",
        spaceAfter=size * 0.35,
    )


def draw_keep(c, items, x, y, w, h, mode="shrink"):
    frame = KeepInFrame(w, h, items, mode=mode, hAlign="LEFT", vAlign="TOP")
    frame.wrapOn(c, w, h)
    frame.drawOn(c, x, y)


def p(text, size=18, leading=None, bold=False, align=TA_LEFT, color=TEXT):
    return Paragraph(text, para_style(size, leading, bold, align, color))


def section_title(text):
    return p(text, 28, 34, True, color=colors.HexColor("#003b77"))


def rounded_box(c, x, y, w, h, fill=None):
    if fill:
        c.setFillColor(fill)
        c.roundRect(x, y, w, h, 14, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setStrokeColor(BLUE)
    c.setLineWidth(4)
    c.roundRect(x, y, w, h, 14, fill=0, stroke=1)


def rect_box(c, x, y, w, h, fill=None, line_width=4):
    if fill:
        c.setFillColor(fill)
        c.rect(x, y, w, h, fill=1, stroke=0)
    c.setStrokeColor(BLUE)
    c.setLineWidth(line_width)
    c.rect(x, y, w, h, fill=0, stroke=1)


def draw_header(c):
    c.setFillColor(TEXT)
    c.setFont(FONT_B, 34)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 60, "2026 1학기 Art & Science 창의융합 오디세이 [창의융합대전]")

    y = PAGE_H - 368
    h = 286
    rect_box(c, 18, y, 1182, h)
    rect_box(c, 18, y, 145, h)
    rect_box(c, 163, y, 145, h)
    rect_box(c, 308, y, 892, h)
    c.line(18, y + 205, 308, y + 205)

    c.setFillColor(TEXT)
    c.setFont(FONT_B, 30)
    c.drawCentredString(90, y + 235, "1분야")
    c.setFont(FONT_B, 21)
    c.drawCentredString(90, y + 205, "[창의융합인]")
    c.setFont(FONT_B, 50)
    c.drawCentredString(90, y + 102, "TE")

    c.setFont(FONT_B, 30)
    c.drawCentredString(235, y + 235, "2분야")
    c.setFont(FONT_B, 23)
    c.drawCentredString(235, y + 207, "(자율)")
    c.setFont(FONT_B, 50)
    c.drawCentredString(235, y + 102, "S")

    title_items = [
        p("EmoNet: 감정을 trace로 표현하는 대화형 AI", 46, 58, True, TA_CENTER),
        Spacer(1, 52),
        p("이은세(1308)", 26, 32, True, TA_CENTER),
    ]
    draw_keep(c, title_items, 325, y + 42, 850, 200)


def draw_main_boxes(c):
    left_x, right_x = 18, 618
    top_y, box_h, box_w = 632, 430, 582
    rounded_box(c, left_x, top_y, box_w, box_h, LIGHT_BLUE)
    rounded_box(c, right_x, top_y, box_w, box_h, LIGHT_GREEN)

    left_items = [
        section_title("1분야 TE: 시모어 패퍼트와 컴퓨팅 기반 사고 확장"),
        p(
            "시모어 패퍼트는 Logo 프로그래밍 언어와 터틀 로봇을 설계한 공학자로, "
            "컴퓨터가 단순 계산기가 아니라 인간의 사고와 문제 해결을 확장하는 도구가 될 수 있음을 보였다.",
            20,
            28,
        ),
        p(
            "그의 핵심 관점은 학습자가 직접 조작하고 실험하면서 개념을 구성한다는 것이다. "
            "즉, 컴퓨팅은 머릿속 과정을 밖으로 꺼내 관찰하고 수정하게 하는 표현 도구가 된다.",
            20,
            28,
        ),
        p(
            "EmoNet도 이 관점과 연결된다. 감정을 “기쁨”, “분노” 같은 최종 라벨로만 출력하지 않고, "
            "입력 자극이 내부 구조를 통과하며 감정 상태를 형성하는 과정을 trace로 표현한다.",
            20,
            28,
        ),
    ]
    draw_keep(c, left_items, left_x + 28, top_y + 28, box_w - 56, box_h - 56)

    right_items = [
        section_title("2분야 S: 감정은 변화하는 내부 상태"),
        p(
            "과학적으로 감정은 한 단어로 즉시 결정되는 결과가 아니라, 자극 해석, 기억, 통제감, "
            "사회적 관계, 행동 경향 등이 함께 작용하며 형성되는 과정이다.",
            20,
            28,
        ),
        p(
            "같은 말이라도 상황에 따라 불안, 분노, 서운함, 방어 반응이 다르게 나타날 수 있다. "
            "따라서 감정을 연구하려면 최종 라벨뿐 아니라 감정이 시간에 따라 어떻게 발생하고 유지되는지 관찰해야 한다.",
            20,
            28,
        ),
        p(
            "EmoNet은 episode를 stim_vec로 바꾸고, node activation, signal propagation, trace, "
            "dominant branch를 통해 감정 흐름을 계산한다.",
            20,
            28,
        ),
    ]
    draw_keep(c, right_items, right_x + 28, top_y + 28, box_w - 56, box_h - 56)

    bottom_x, bottom_y, bottom_w, bottom_h = 18, 34, 1182, 572
    rounded_box(c, bottom_x, bottom_y, bottom_w, bottom_h, colors.white)
    bottom_items = [
        section_title("융합 탐구: EmoNet trace는 AI의 감정 상태를 설명 가능하게 표현할 수 있는가?"),
        p(
            "<b>탐구 주제</b> - 대화 상황을 입력했을 때 EmoNet의 trace가 단순 감정 라벨보다 "
            "감정 상태의 형성 과정과 반응 방향을 더 잘 설명하는지 확인한다.",
            19,
            26,
        ),
        p(
            "<b>탐구 방법</b> - 1) 감정을 유발하는 대화 상황 episode를 입력한다. "
            "2) EmoNet이 episode를 stim_vec로 변환한다. "
            "3) stim_vec가 내부 node를 활성화하며 시간 순서의 trace를 만든다. "
            "4) trace에서 dominant branch를 추출해 핵심 감정 흐름을 확인한다. "
            "5) 감정 라벨만 사용하는 방식과 trace를 사용하는 방식을 비교한다.",
            19,
            26,
        ),
        p(
            "<b>탐구 결과</b> - EmoNet은 감정을 단일 라벨로 바로 정하지 않고, 입력 자극이 내부에서 "
            "어떤 경로로 확산되고 유지되는지를 trace로 기록한다. 이 trace에는 활성 강도, 지속성, "
            "branch 흐름, 감정축 변화가 포함되어 감정 상태가 만들어지는 과정을 볼 수 있다.",
            19,
            26,
        ),
        p(
            "부정적 대화 상황에서는 위협, 관계 손상, 통제감 저하와 관련된 흐름이 활성화될 수 있으며, "
            "dominant branch는 최종적으로 AI가 어떤 정서적 방향으로 반응할 가능성이 큰지를 보여준다.",
            19,
            26,
        ),
        p(
            "<b>결론</b> - EmoNet은 패퍼트의 컴퓨팅 기반 사고 확장 관점과 감정 과학을 융합한 탐구이다. "
            "컴퓨터 모델을 이용해 감정을 단순 분류 결과가 아니라 시간에 따라 변화하는 내부 trace로 표현함으로써, "
            "대화형 AI의 감정 반응을 더 설명 가능하고 실험 가능한 형태로 만들 수 있다.",
            19,
            26,
        ),
    ]
    draw_keep(c, bottom_items, bottom_x + 32, bottom_y + 34, bottom_w - 64, bottom_h - 68)


def main():
    c = canvas.Canvas(str(PDF_PATH), pagesize=(PAGE_W, PAGE_H))
    c.setTitle("EmoNet 창의융합 오디세이 포스터")
    draw_header(c)
    draw_main_boxes(c)
    c.showPage()
    c.save()
    print(PDF_PATH)


if __name__ == "__main__":
    main()
