from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routers.chat_router import chat_router
from app.services.faq_services import FAQService


# FAQService 인스턴스 생성
faq_service = FAQService()

# Milvus 초기화 및 데이터 로딩
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan 이벤트를 처리하는 함수.
    """
    if faq_service.is_initialized():  # Milvus 컬렉션이 비어 있는지 확인
        print("✅ Milvus 컬렉션이 이미 초기화되어 있습니다. 데이터를 로드하지 않습니다.")
    else:
        await faq_service.load_and_store_pkl()
    yield  # FastAPI가 lifespan 이벤트를 처리할 수 있도록 함
    
    print("🛑 Lifespan 종료: 리소스 정리 완료")

# FastAPI 애플리케이션 생성
app = FastAPI(lifespan=lifespan)

# 라우터 등록
app.include_router(chat_router, prefix="/chat", tags=["Chat"])

