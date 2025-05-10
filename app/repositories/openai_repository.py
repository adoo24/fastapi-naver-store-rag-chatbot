import openai
from app.config import OPENAI_API_KEY
import asyncio

class OpenAIRepository:
    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def generate_embedding(self, text: str):
        """
        OpenAI를 사용하여 텍스트를 임베딩으로 변환
        """
        response = await self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    async def refine_question(self, session_context: str, question: str) -> str:
        """
        이전 대화 맥락을 바탕으로 질문을 정제합니다.
        :param session_context: 이전 대화 맥락
        :param question: 사용자가 입력한 질문
        :return: 정제된 질문
        """
        print(f"정제할 질문: {question}")  # 디버깅용 출력
        print(f"세션 맥락: {session_context}")
        if not session_context:
            return question
        prompt = f"""
            아래는 사용자의 이전 대화 맥락입니다. 이 맥락을 바탕으로 사용자의 질문하고자 하는 의도와 궁금해 하는 것을 파악하고, 명확하고 간결하게 정제해주세요.
            주어와 목적어를 명확히 하고, 불필요한 부분은 제거해주세요
        맥락:
        {session_context}

        현재 질문: {question}

        정제된 질문:
        """
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()

    async def stream_answer_question(self, refined_question: str, context: str):
        """
        OpenAI GPT 모델을 사용하여 SSE 방식으로 응답을 스트리밍합니다.
        :param refined_question: 정제된 질문
        :param context: 질문에 대한 추가 맥락
        """
        prompt = f"""
            너는 네이버 스마트스토어 관련 질문에 답변하는 스마트하고 친절한 상담원 역할을 맡고 있어.
            사용자는 네이버 스마트스토어에 대해 궁금한 점을 질문하는 사람이며, 정확하고 신뢰도 높은 답변을 기대하고 있어.

            📌 **답변 규칙:**
            1. 질문에 대한 답변은 **반드시 네이버 스마트스토어 FAQ의 정보**를 기반으로 작성하세요.
            2. **FAQ에 없는 내용은 포함하지 마세요.** 사실이 확인되지 않은 내용이나, FAQ에서 언급되지 않은 정보는 제공하지 않습니다.
            3. **FAQ에서 확인된 정보는 자연스럽고 친절한 대화체로 설명**하세요.
            4. **번역투 어투는 피하고, 자연스러운 한국어 대화체**로 꼭 답변하세요.
            5. **사용자의 이전 질문과 답변을 고려하여, 맥락을 이해하고 이어지는 질문을 처리**하세요.
            6. **현재 질문에 대한 답변을 제공한 뒤, 이전 질문들과 연결하여 사용자 상황을 고려한 추가 질문 2개**를 제안하세요.
            7. 만약 사용자의 질문이 네이버 스마트스토어와 무관하거나 FAQ에서 찾을 수 없는 질문일 경우:
            - "저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다." 라고 안내하세요.
            - 그런 질문과 관련된 FAQ가 있다면 관련된 질문을 제안하세요.

            📌 **대화 맥락 예제:**
            - 사용자 질문 1: "스마트스토어에서 상품을 어떻게 등록하나요?"
            - 챗봇 응답: 
            스마트스토어에서 상품을 등록하는 방법은 다음 단계로 진행할 수 있습니다.  
            1. 스마트스토어 관리자 페이지에 로그인하세요.  
            2. 상품 관리 메뉴에서 "상품 등록"을 선택하세요.  
            3. 상품 정보를 입력하고, 저장하여 등록을 완료하세요.  

            혹시 상품을 수정하거나 삭제하는 방법도 궁금하신가요?  
            또는 등록한 상품의 노출 방법도 안내드릴까요?  

            **현재 사용자 질문: {refined_question}**
            **이전 질문들:**
            {context}

            **답변:**

        """
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            stream=True,  # 스트리밍 활성화
            max_tokens=500,
            temperature=0.7
        )

        # 스트리밍 응답 처리
        async for chunk in response:
            choice = chunk.choices[0].delta.content
            if choice:
                yield f"data: {choice}"
    async def extract_keyword(self, question: str) -> list[str] :
        """
        질문에서 핵심 키워드 리스트를 추출합니다.
        :param question: 사용자가 입력한 질문
        :return: 추출된 키워드 리스트
        """
        prompt = f"""
            너는 질문에서 중요한 핵심 키워드 구문을 추출하는 역할을 맡고 있어. 
            질문에서 명사, 주요 동사, 중요한 개념을 조합하여 핵심 키워드 구문을 정확히 추출해줘.

            아래는 몇 가지 규칙이야:
            1. 각 질문에서 의미를 나타내는 핵심 키워드 구문 (2~3개 단어)을 추출해.
            2. 동의어나 유사어는 하나의 키워드 구문으로 통합해.
            3. 너무 일반적인 단어 (예: "어떻게", "할 수 있나요")는 제외해.
            4. 가능한 한 구체적이고 의미가 분명한 키워드 구문을 선택해.
            5. 결과는 쉼표로 구분된 리스트 형태로 제공해.

            예제:
            질문: "스마트스토어에서 상품을 등록하는 방법이 뭐에요?"
            스마트스토어 상품 등록, 등록 방법

            질문: "환불 정책은 어떻게 적용되나요?"
            환불 정책, 정책 적용

            질문: "{question}"
            

        """
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.5
        )
        keywords = response.choices[0].message.content.strip().split(",")
        return [keyword.strip() for keyword in keywords]
