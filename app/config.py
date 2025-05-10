import os

# OpenAI API 키
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")  # 환경 변수에서 가져오거나 기본값 사용

# Milvus 설정
MILVUS_HOST = "localhost"  # Milvus 서버 호스트
MILVUS_PORT = "19530"     # Milvus 서버 포트