import re
from pymilvus import connections, Collection

# Milvus 서버 연결
def init_milvus():
    connections.connect("default", host="localhost", port="19530")
    collection = Collection("faq_collection")
    collection.load()
    return collection

# 텍스트 클리닝 함수 (Answer 필드 전용)
def clean_text(text):
    """
    Answer 필드의 텍스트를 클리닝하는 함수.
    """
    text = re.sub(r'[\x00-\x1F\x7F-\x9F\xa0]', ' ', text)  # 제어 문자 및 \xa0 제거
    text = re.sub(r'&nbsp;|&quot;|&amp;|&lt;|&gt;', ' ', text)  # HTML 엔티티 제거
    text = re.sub(r'위 도움말이 도움이 되었나요.*(별점[0-9]점)+.*?소중한 의견을 남겨주시면 보완하도록 노력하겠습니다.*', '', text, flags=re.DOTALL)
    text = re.sub(r'\s+', ' ', text).strip()  # 중복된 공백 정리
    return text

# Milvus에서 Answer 필드 클리닝
def clean_answers_in_milvus(batch_size=100):
    collection = init_milvus()
    
    # 전체 데이터 개수 확인
    total_count = collection.num_entities
    print(f"📊 총 데이터 개수: {total_count}")
    
    updated_records = []

    # 데이터를 batch_size 단위로 로드
    for offset in range(0, total_count, batch_size):
        all_data = collection.query(
            expr="",
            output_fields=["question", "answer", "embedding"],
            limit=batch_size,
            offset=offset
        )
        
        for record in all_data:
            question = record["question"]
            answer = clean_text(record["answer"])
            embedding = record["embedding"]

            updated_records.append([question, answer, embedding])
            print(f"✅ 클리닝 완료: {question} -> {answer}")
    
    # 기존 데이터 삭제 (모든 데이터 삭제)
    collection.delete(expr="question != ''")  # 모든 데이터를 삭제하는 조건식
    
    # 클리닝된 데이터 다시 삽입
    collection.insert([[rec[0] for rec in updated_records], 
                       [rec[1] for rec in updated_records], 
                       [rec[2] for rec in updated_records]])
    print(f"✅ 모든 Answer 필드 클리닝 완료 (총 {len(updated_records)}개)")

clean_answers_in_milvus(batch_size=100)
