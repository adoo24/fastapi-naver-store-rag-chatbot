import re

def clean_text(text: str) -> str:
    """
    텍스트 클리닝 함수
    """
    text = re.sub(r'[\x00-\x1F\x7F-\x9F\xa0]', ' ', text)  # 제어 문자 및 \xa0 제거
    text = re.sub(r'&nbsp;|&quot;|&amp;|&lt;|&gt;', ' ', text)  # HTML 엔티티 제거
    text = re.sub(r'\s+', ' ', text).strip()  # 중복된 공백 정리
    return text