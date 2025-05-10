from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.services.faq_services import FAQService

# 라우터 생성
chat_router = APIRouter()

# FAQService 인스턴스 생성
faq_service = FAQService()

@chat_router.get("/", response_class=StreamingResponse)
async def chat(question: str, session_id: str = ""):
    """
    Chat 엔드포인트: 질문에 대한 실시간 응답을 반환합니다.
    """
    if not question:
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")

    return StreamingResponse(faq_service.answer_question(question, session_id), media_type="text/event-stream")