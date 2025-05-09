from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse
from app.chat import answer_question, stream_answer_question
from app.services.faq_services import FAQService
from app.embeddings import init_milvus, load_and_store_pkl
import os
import logging

logger = logging.getLogger("uvicorn")

# Milvus 초기화 및 데이터 로딩
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan 이벤트를 처리하는 함수.
    """
    collection = init_milvus()
    if False:  # Milvus 컬렉션이 비어 있는지 확인
        file_path = "final_result.pkl"  # .pkl 파일 경로
        if os.path.exists(file_path):
            load_and_store_pkl(file_path)
        else:
            print(f"❌ {file_path} 파일이 존재하지 않습니다. 데이터를 로드할 수 없습니다.")
    else:
        print("✅ Milvus 컬렉션이 이미 초기화되어 있습니다. 데이터를 로드하지 않습니다.")
    
    # Lifespan 시작
    yield  # FastAPI가 lifespan 이벤트를 처리할 수 있도록 함
    
    # Lifespan 종료
    print("🛑 Lifespan 종료: 리소스 정리 완료")

app = FastAPI(lifespan=lifespan)
# FAQService 인스턴스 생성
faq_service = FAQService()

@app.get("/chat", response_class=StreamingResponse)
async def chat(question: str, session_id: str = ""):
    if not question:
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")

    return StreamingResponse(faq_service.answer_question(question, session_id), media_type="text/event-stream")
