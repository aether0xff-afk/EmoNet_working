[TASK]
주어진 대화 입력에 대해 어울리는 응답 스타일 벡터 `s` 와 예시 응답 1개를 생성하라.

[INPUT_TEXT]
{{input_text}}

[LATENT_Z]
{{z_lines}}

[STYLE_AXES]
1. verbosity
2. sentence_length
3. pace
4. fragmentation
5. repetition
6. rhythmicity
7. directness
8. explicitness
9. specificity
10. abstraction
11. certainty
12. logicality
13. warmth
14. distance
15. politeness
16. formality
17. cooperativeness
18. dominance
19. calmness
20. tension
21. positivity
22. heaviness
23. urgency
24. emotional_openness
25. softness
26. sharpness
27. playfulness
28. seriousness
29. metaphoricity
30. plainness
31. initiative
32. reflectiveness

[OUTPUT_FORMAT]
JSON only.

```json
{
  "s": {
    "verbosity": 0.0,
    "sentence_length": 0.0,
    "pace": 0.0,
    "fragmentation": 0.0,
    "repetition": 0.0,
    "rhythmicity": 0.0,
    "directness": 0.0,
    "explicitness": 0.0,
    "specificity": 0.0,
    "abstraction": 0.0,
    "certainty": 0.0,
    "logicality": 0.0,
    "warmth": 0.0,
    "distance": 0.0,
    "politeness": 0.0,
    "formality": 0.0,
    "cooperativeness": 0.0,
    "dominance": 0.0,
    "calmness": 0.0,
    "tension": 0.0,
    "positivity": 0.0,
    "heaviness": 0.0,
    "urgency": 0.0,
    "emotional_openness": 0.0,
    "softness": 0.0,
    "sharpness": 0.0,
    "playfulness": 0.0,
    "seriousness": 0.0,
    "metaphoricity": 0.0,
    "plainness": 0.0,
    "initiative": 0.0,
    "reflectiveness": 0.0
  },
  "response": "string"
}
```

[CONSTRAINTS]
- `s` 는 32축 모두 0~1 범위 실수로 채운다.
- `response` 는 입력 내용과 정합적이어야 한다.
- 자연스러운 한국어를 사용한다.
- `z` 는 직접 언급하지 말고 내부 상태 힌트로만 사용한다.
