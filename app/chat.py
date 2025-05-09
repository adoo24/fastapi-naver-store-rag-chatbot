import openai
import uuid  # UUID를 사용하여 고유한 session_id 생성
from app.embeddings import search_faq
from app.config import OPENAI_API_KEY  # config.py에서 API 키 가져오기
from typing import Dict, List

# 세션 데이터 저장소 (메모리 기반, 필요시 DB로 변경 가능)
session_data: Dict[str, List[Dict[str, str]]] = {}

# 슬라이딩 윈도우 크기 설정 (최대 저장 가능한 대화 이력 개수)
WINDOW_SIZE = 10

# 사용자 메시지 저장 (슬라이딩 윈도우 방식)
def save_user_message(session_id: str, question: str, response: str):
    if session_id not in session_data:
        session_data[session_id] = []
    session_data[session_id].append({"question": question, "response": response})
    
    # 슬라이딩 윈도우 적용: 최대 WINDOW_SIZE 개수만 유지
    if len(session_data[session_id]) > WINDOW_SIZE:
        session_data[session_id] = session_data[session_id][-WINDOW_SIZE:]

# 중요도 기반 맥락 생성
def get_important_context(session_id: str) -> str:
    if session_id not in session_data:
        return ""
    # 중요도가 높은 최근 3개의 질문과 응답을 포함
    important_messages = session_data[session_id][-3:]
    return "\n".join([f"User: {msg['question']}\nBot: {msg['response']}" for msg in important_messages])

# 질문 정제 (이전 맥락 기반으로 현재 질문을 정제)
def refine_question(session_id: str, question: str) -> str:
    if session_id not in session_data:
        return question  # 이전 맥락이 없으면 원래 질문 반환

    # 이전 질문과 응답을 맥락으로 생성
    context = "\n".join([f"User: {msg['question']}\nBot: {msg['response']}" for msg in session_data[session_id]])
    prompt = f"""
    아래는 사용자의 이전 대화 맥락입니다. 이 맥락을 바탕으로 사용자의 현재 질문을 명확하고 간결하게 정제해주세요.

    맥락:
    {context}

    현재 질문: {question}

    정제된 질문:
    """
    client = openai.Client(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that refines user questions based on context."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0.5
    )
    refined_question = response.choices[0].message.content.strip()
    print(f"정제된 질문: {refined_question}")  # 디버깅용 출력
    return refined_question

# 질문에 대한 응답 생성
async def answer_question(session_id: str = None, question: str = "") -> dict:
    # session_id가 없으면 기본값 생성
    if not session_id:
        session_id = str(uuid.uuid4())  # 고유한 session_id 생성

    # 질문 정제
    refined_question = refine_question(session_id, question)

    # Milvus에서 유사 질문 검색
    related_questions = search_faq(refined_question)
    
    # FAQ를 컨텍스트에 포함
    faq_context = "\n\n".join([f"- {q}" for q in related_questions])
    print(f"유사 질문 목록: {faq_context}")  # 디버깅용 출력

    # 중요도 기반 맥락 생성
    important_context = get_important_context(session_id)
    final_context = f"{important_context}\n\nFAQ:\n{faq_context}"

    # GPT로 응답 생성
    prompt = f"""
    너는 네이버 스마트스토어 관련 질문에 답변하는 직원이야.
    아래는 네이버 스마트스토어 관련 자주 묻는 질문(FAQ)에서 추출된 정보입니다.
    이를 바탕으로 사용자 질문에 신뢰도 높은 답변을 제공해주세요.
    사용자의 질문에 대해 답을 해준 뒤, 질의응답 맥락에서 사용자가 궁금해할만한 다른 내용을 2개 추가로 물어봐야 합니다.
    
    번역 어투는 피하고, 자연스러운 대화체로 작성해주세요.
    질문에 대한 답변은 FAQ의 내용을 바탕으로 작성하되, FAQ에 없는 내용은 포함하지 마세요, 정보가 부족하다고 판단할 시에는 "정보가 부족합니다."라고 답변해주세요.
    그 이외의 질문이 들어오면 "저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다."와 같이 안내 메시지를 주세요.

    맥락: {final_context}

    사용자 질문: {refined_question}
    답변:
    """
    print(f"최종 맥락:\n{final_context}")  # 디버깅용 출력

    # OpenAI API 키 확인
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. config.py 또는 환경 변수를 확인하세요.")
    
    # OpenAI 클라이언트 생성
    client = openai.Client(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant providing accurate answers based on the given information. Avoid mentioning that you are referring to FAQs."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7
    )
    bot_response = response.choices[0].message.content.strip()
    
    # 응답 저장 (슬라이딩 윈도우 적용)
    save_user_message(session_id, question, bot_response)
    
    return {
        "session_id": session_id,  # 생성된 session_id 반환
        "answer": bot_response,
        "related_questions": related_questions  # 관련 질문도 반환
    }
