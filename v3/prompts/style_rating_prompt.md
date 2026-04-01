[TASK]
아래 입력과 응답을 보고 응답의 스타일 벡터 `s_hat` 를 32축 0~1 값으로 평가하라.

[INPUT_TEXT]
{{input_text}}

[RESPONSE]
{{response}}

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
  "s_hat": {
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
  "notes": "short string"
}
```

[CONSTRAINTS]
- 내용 적합성보다 응답의 말투와 표현 특성만 평가한다.
- 응답에 표정 변화 같은 명시적 비언어 단서가 있으면 그것도 표현 특성으로 반영한다.
- 각 축은 반드시 0~1 범위로 준다.
- `notes` 는 한 문장으로 짧게 쓴다.
