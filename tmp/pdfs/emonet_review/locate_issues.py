from pathlib import Path
text=Path('tmp/pdfs/emonet_review/extracted.txt').read_text(encoding='utf-8', errors='replace')
queries=['ㅋ;워드','고려 하지','생성 함','활성화 되기','학습할때','무기력함에 끊기게','표X','그림X','그림 X','데이터 이기 떄문','살아 남','활성 화','한다.7. 결과','8.1 후속 연구','3. 선행 연구 분석','4. EmoNet의 구성']
for pno,page_text in enumerate(text.split('\f'),1):
    for lno,line in enumerate(page_text.splitlines(),1):
        for q in queries:
            if q in line:
                print(f'p{pno} l{lno}: {line.strip()}')
