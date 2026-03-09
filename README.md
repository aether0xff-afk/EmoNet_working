1. ML 방식 비교
2. LLM 방식 비교
3. ML, LLM 중 확실한거 고르기
4. encoder 완성


**PIPELINE**

*input(x) -> Encoder(E) -> stimulus_vec(u) -> nn -> H(emo_hist) -> H_encoder -> z(latent_emo_vec) -> style_mapper -> style(s) -> Decoder -> output*


* (x): 입력 텍스트
* (E): 인코더
* (u): 자극 벡터
* (nn): 감정 동역학 신경망
* (H): 감정 히스토리
* (z): 히스토리를 압축한 잠재 상태
* (s): 최종 응답 스타일/정서 상태 벡터
* (Decoder): 디코더 또는 출력 생성기
* (output): 최종 출력 텍스트


