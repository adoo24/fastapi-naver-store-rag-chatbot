import os
import pickle
import numpy as np
from pymilvus import connections, Collection, CollectionSchema, DataType, FieldSchema, Index
import openai
import json
import re
from app.config import OPENAI_API_KEY  # config.py에서 API 키 가져오기

def init_milvus():
    connections.connect("default", host="localhost", port="19530")
    collection_name = "faq_collection"
    
    # 새 컬렉션 생성
    fields = [
        FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=500, is_primary=True),
        FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=65535),  # 응답 필드 추가
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)
    ]
    schema = CollectionSchema(fields=fields, description="FAQ Collection")
    
    collection = Collection(collection_name, schema=schema, using="default")
    
    # 인덱스 생성
    if not collection.has_index():
        index_params = {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}}
        collection.create_index(field_name="embedding", index_params=index_params)
        print("✅ 인덱스 생성 완료")
    
    collection.load()
    return collection

# 질문 벡터화 및 Milvus 저장
def embed_and_store_question(question: str):
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. config.py 또는 환경 변수를 확인하세요.")
    
    client = openai.Client(api_key=OPENAI_API_KEY)  # OpenAI 클라이언트 생성
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    embedding = response.data[0].embedding  # 객체 속성으로 접근
    
    collection = init_milvus()
    collection.insert([[question], [embedding]])
    print(f"✅ 질문 저장 완료: {question}")

# 텍스트 클리닝 함수 (질문/답변)
def clean_text(text):
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)  # 제어 문자 제거
    text = re.sub(r'&nbsp;|&quot;|&amp;|&lt;|&gt;', ' ', text)  # HTML 엔티티 제거
    text = re.sub(r'위 도움말이 도움이 되었나요\?.*?(별점[0-9]점)+.*?소중한 의견을 남겨주시면 보완하도록 노력하겠습니다\.', '', text, flags=re.DOTALL)
    text = re.sub(r'\s+', ' ', text).strip()  # 중복된 공백 정리
    return text

def load_and_store_pkl(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} 파일이 존재하지 않습니다.")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. config.py 또는 환경 변수를 확인하세요.")
    
    collection = init_milvus()
    client = openai.Client(api_key=OPENAI_API_KEY)
    
    # 각 질문을 벡터화하여 Milvus에 저장
    for question, answer in data.items():
        # 질문 문자열을 JSON 형식으로 이스케이프 처리
        escaped_question = json.dumps(question)
        
        # 중복 삽입 방지 (예: 질문이 이미 존재하는지 확인)
        search_results = collection.query(
            expr=f'question == {escaped_question}',
            output_fields=["question"]
        )
        if search_results:
            print(f"⚠️ 이미 존재하는 질문: {question}")
            continue

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=question
        )
        embedding = response.data[0].embedding
        
        # Milvus에 질문, 응답, 벡터 저장
        collection.insert([[question], [answer], [embedding]])
        print(f"✅ 질문과 응답 저장 완료: {question} -> {answer}")

def search_faq(question: str):
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. config.py 또는 환경 변수를 확인하세요.")
    
    client = openai.Client(api_key=OPENAI_API_KEY)  # OpenAI 클라이언트 생성
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    embedding = response.data[0].embedding  # 객체 속성으로 접근
    
    collection = init_milvus()
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    
    # 벡터 검색
    results = collection.search(
        data=[embedding],
        anns_field="embedding",
        param=search_params,
        limit=3,
        output_fields=["question", "answer"]  # 응답 필드 포함
    )
    
    # 검색 결과 반환 (질문과 응답 모두 포함)
    return [{"question": hit.entity.get("question"), "answer": hit.entity.get("answer")} for hit in results[0]]