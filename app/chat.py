import openai
from app.embeddings import search_faq
from app.config import OPENAI_API_KEY  # config.py에서 API 키 가져오기

async def answer_question(question: str) -> dict:
    # Milvus에서 유사 질문 검색
    related_questions = search_faq(question)
    
    # GPT로 응답 생성 (자연스럽고 신뢰도 높은 답변)
    context = "\n\n".join([f"- {q}" for q in related_questions])  # 질문 목록 포맷팅
    print(f"유사 질문 목록: {context}")  # 디버깅용 출력
    prompt = f"""
    아래는 네이버 스마트스토어 관련 자주 묻는 질문(FAQ)에서 추출된 정보입니다.
    이를 바탕으로 사용자 질문에 신뢰도 높은 답변을 제공해주세요.

    FAQ 내용: {context}

    사용자 질문: {question}
    답변:
    """

  
    
    # OpenAI 클라이언트 생성
    client = openai.Client(api_key=OPENAI_API_KEY)
    
    # Chat Completion 생성
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant providing accurate answers based on the given information. Avoid mentioning that you are referring to FAQs."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7
    )
    
    # 응답 내용 반환
    answer = response.choices[0].message.content.strip()
    return {
        "answer": answer,
        "related_questions": related_questions  # 관련 질문도 반환
    }
