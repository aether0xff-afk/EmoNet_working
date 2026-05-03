# Human Evaluation Instructions

각 행은 하나의 사용자 입력과 여러 후보 응답으로 구성된다.
후보 열은 candidate_a, candidate_b 순서로 제시되며, 어떤 모델 조건인지 숨겨져 있다.

권장 평가 항목:

- content_fit: 입력 내용에 직접적으로 맞는가
- emotional_appropriateness: 입력 감정 상태에 맞는가
- style_match: 더 설득력 있는 말투를 보이는가
- naturalness: 한국어 응답이 자연스러운가

실제 평가 시에는 각 항목별 점수 또는 최고 후보를 별도 시트에 기록한다.