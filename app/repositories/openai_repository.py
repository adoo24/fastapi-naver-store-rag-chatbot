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
            너는 네이버 스마트스토어 관련 질문에 답변하는 직원이야.
            사용자는 네이버 스마트스토어에 대해서 궁금해하는 사람이야.
            아래는 네이버 스마트스토어 관련 자주 묻는 질문(FAQ)에서 추출된 정보입니다.
            이를 바탕으로 사용자 질문에 신뢰도 높은 답변을 제공해주세요.
            사용자의 질문에 대해 답을 해준 뒤, 너의 답변에서 사용자가 처한 상황을 고려해서 궁금해할만한 다른 내용을 2개를 개행을 두어서 물어봐야 합니다.
            
            번역 어투는 피하고, 자연스러운 한국어 대화체로 꼭 !!! 작성해주세요.
            질문에 대한 답변은 FAQ의 내용을 바탕으로 작성하되, FAQ에 없는 내용은 포함하지 마세요
            그 이외의 질문이 들어오면 "저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 요탁드립니다."와 같이 안내 메시지를 주세요.

        맥락:
        {context}

        사용자 질문: {refined_question}

        답변:
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
                print(f"스트리밍 응답: {choice}")
                yield f"data: {choice}"
