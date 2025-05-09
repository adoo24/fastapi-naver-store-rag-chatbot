from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from app.chat import answer_question
from app.embeddings import init_milvus, load_and_store_pkl
import os
import logging

app = FastAPI()
logger = logging.getLogger("uvicorn")

# Milvus 초기화 및 데이터 로딩
@app.on_event("startup")
async def startup_event():
    collection = init_milvus()
    if False:  # Milvus 컬렉션이 비어 있는지 확인
        file_path = "final_result.pkl"  # .pkl 파일 경로
        if os.path.exists(file_path):
            load_and_store_pkl(file_path)
        else:
            print(f"❌ {file_path} 파일이 존재하지 않습니다. 데이터를 로드할 수 없습니다.")
    else:
        print("✅ Milvus 컬렉션이 이미 초기화되어 있습니다. 데이터를 로드하지 않습니다.")


# SSE 스트리밍 생성기
async def sse_stream(question: str):
    try:
        response = await answer_question("default", question)
        yield f"data: 질문: {question}\n\n"
        yield f"data: 응답: {response['answer']}\n\n"
        for related_question in response["related_questions"]:
            yield f"data: 관련 질문: {related_question}\n\n"
        yield "data: [END]\n\n"
    except Exception as e:
        yield f"data: 오류 발생: {str(e)}\n\n"

@app.get("/chat", response_class=StreamingResponse)
async def chat(question: str):
    if not question:
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")
    
    return StreamingResponse(sse_stream(question), media_type="text/event-stream")
