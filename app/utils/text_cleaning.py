import re

def clean_text(text):
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)  # 제어 문자 제거
    text = re.sub(r'&nbsp;|&quot;|&amp;|&lt;|&gt;', ' ', text)  # HTML 엔티티 제거
    text = re.sub(r'위 도움말이 도움이 되었나요\?.*?(별점[0-9]점)+.*?소중한 의견을 남겨주시면 보완하도록 노력하겠습니다\.', '', text, flags=re.DOTALL)
    text = re.sub(r'\s+', ' ', text).strip()  # 중복된 공백 정리
    return text