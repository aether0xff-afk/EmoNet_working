# Figure Prompts for Section 5

## Figure 5-1. EmoNet Overall Architecture

A clean academic diagram in Korean showing the overall architecture of EmoNet.

Left to right flow:
"episode" -> "stim_vec (4차원 자극 벡터)" -> "256-node neural network" -> "tick별 trace" -> "branch extraction" -> "dominant branch" -> "cluster path" -> "semantic report".

Use simple boxes and arrows. Add section labels above the flow:
"입력부", "신경망 처리부", "기록부", "구조화부", "해석부".

Style: white background, blue accent color, gray secondary lines, minimal academic layout, paper-friendly, high readability, no 3D, no decorative background.

## Figure 5-2. Episode to Stim Vec

A clean academic Korean diagram explaining how an "episode" is converted into a 4-dimensional "stim_vec".

On the left, a rounded box labeled "episode" with example text:
"친구가 사람들 앞에서 나를 무시했다".

In the middle, show extracted cues:
"공개적 모욕", "사회적 위협", "부당함 판단", "관계 손상 가능성", "반격 충동", "표현 억제".

On the right, show a vector box labeled "stim_vec" with four abstract dimensions:
"d1", "d2", "d3", "d4".

Use arrows from episode to cues to vector. Style: minimal academic style, white background, blue and gray accents, Korean labels, high readability.

## Figure 5-3. One Tick Activation

A clean Korean academic flowchart showing one tick of node activation in EmoNet.

Flow:
"현재 stim_vec" + "이전 기억(memory)" + "연결 노드 입력" - "피로도/억제" -> "활성 후보값 K 계산" -> "threshold 비교".

From "threshold 비교", split into two paths:
1. pass: "노드 활성화 및 신호 전달"
2. fail: "약화 또는 비활성"

Use a simple decision diamond for threshold. Style: white background, blue accent, gray secondary lines, readable Korean text, no 3D.

## Figure 5-4. Trace Time-Series

A clean academic diagram in Korean showing trace as a time-series record.

Create a horizontal timeline labeled:
t0, t1, t2, t3, t4, t5.

At each time point, show small node circles with different activation strengths using blue intensity. Connect selected nodes across time with thin arrows to show internal flow.

Under the timeline, add the label:
"trace = tick별 내부 활성 기록".

Add small annotations:
"활성 정보", "강도 정보", "지속 정보", "경로 정보".

Style: minimal white background, paper-friendly, blue and gray accents, high readability.

## Figure 5-5. Branch and Dominant Branch

A clean Korean academic network diagram showing branches inside a trace.

Start with node n0. From n0, branch into multiple paths. Use:
- light gray dashed paths for "dropped branch"
- gray paths for "weak branch"
- dark gray paths for "competing branch"
- one thick blue path for "dominant branch"

Label the thick blue path:
"dominant branch".

Include a bottom caption inside the figure:
"dominant branch = 가장 강하고 오래 유지된 대표 내부 경로".

Style: white background, simple circles and arrows, blue accent, high readability, no decorative background.

## Figure 5-6. Node Path to Cluster Path

A clean academic Korean diagram showing conversion from node path to cluster path.

Top row:
node path with circles:
"n0 -> n3 -> n6 -> n10 -> n13".

Bottom row:
cluster path with rounded boxes:
"위협 감지" -> "부당함 평가" -> "반격 충동" -> "표현 준비" -> "최종 상태".

Use vertical mapping arrows from each node to its cluster meaning.

Style: white background, blue and gray accents, minimal academic style, high readability, no 3D.

## Figure 5-7. Emotional Axes and Trace Interpretation

A clean academic Korean diagram showing emotional axes used to interpret trace and branch.

Center: a blue box labeled:
"trace / branch".

Around it, place axis labels:
"감정가(긍정-부정)",
"통제감(높음-낮음)",
"사회적 방향(자기-타인-관계)",
"행동 경향(접근-회피-반격-억제)",
"표현-억제",
"책임 귀인".

Use arrows from the axis labels to the center box.

Style: white background, paper-friendly, simple layout, blue and gray accents, high readability, no decoration.
