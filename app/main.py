from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse
from app.services.faq_services import FAQService


# Milvus 초기화 및 데이터 로딩
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan 이벤트를 처리하는 함수.
    """
    if False:  # Milvus 컬렉션이 비어 있는지 확인
        await faq_service.load_and_store_pkl()
    else:
        print("✅ Milvus 컬렉션이 이미 초기화되어 있습니다. 데이터를 로드하지 않습니다.")
    yield  # FastAPI가 lifespan 이벤트를 처리할 수 있도록 함
    
    print("🛑 Lifespan 종료: 리소스 정리 완료")

app = FastAPI(lifespan=lifespan)
# FAQService 인스턴스 생성
faq_service = FAQService()

@app.get("/chat", response_class=StreamingResponse)
async def chat(question: str, session_id: str = ""):
    if not question:
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")

    return StreamingResponse(faq_service.answer_question(question, session_id), media_type="text/event-stream")
