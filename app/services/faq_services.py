from app.repositories.milvus_repository import MilvusRepository
from app.repositories.openai_repository import OpenAIRepository
from app.repositories.context_repository import ContextRepository

class FAQService:
    def __init__(self):
        """
        FAQService 초기화 시 Milvus 컬렉션을 초기화합니다.
        """
        self.milvus_repo = MilvusRepository()
        self.openai_repo = OpenAIRepository()
        self.context_repo = ContextRepository()  # 사용자 세션 컨텍스트 관리
        self.milvus_repo.initialize()  # Milvus 초기화

    async def answer_question(self, question: str, session_id: str = ""):
        """
        SSE 방식으로 질문에 대한 응답을 스트리밍합니다.
        :param session_id: 사용자 세션 ID
        :param question: 사용자가 입력한 질문
        """
        if not session_id:
            session_id = self.context_repo.create_session()  # 새로운 세션 생성

        # 1. 질문 정제
        session_context = self.context_repo.get_context(session_id)  # 이전 대화 맥락 가져오기
        refined_question = await self.openai_repo.refine_question(session_context, question)
        yield f"data: 정제된 질문: {refined_question}\n\n"

        # 2. Milvus에서 유사 질문 검색
        embedding = await self.openai_repo.generate_embedding(refined_question)  # 질문을 임베딩으로 변환
        if embedding is None:
            raise ValueError("임베딩 생성에 실패했습니다.")
        related_questions = self.milvus_repo.find_similar_faqs(embedding)
        faq_context = "\n\n".join([f"- {q['question']}" for q in related_questions])
        
        if len(related_questions) <= 5:
            # 유사 질문이 부족함, 보강 필요한 질문 저장
            self.context_repo.log_insuffiecient_context_question(refined_question, embedding)
        yield f"data: --- \n유사 질문 목록:\n{faq_context}\n\n --- \n\n"

        # 3. 중요도 기반 맥락 생성
        
        # 4. OpenAI GPT로 응답 생성
        full_answer = ""  # 스트리밍 데이터를 저장할 변수
        async for message in self.openai_repo.stream_answer_question(refined_question, faq_context):
            yield message
            # 스트리밍 데이터를 합쳐서 저장
            if message.startswith("data: "):
                print(f"수신된 메시지: {message[6:]}")  # 디버깅용 출력
                full_answer += message[6:]

        # 5. 세션 데이터 업데이트
        self.context_repo.save_user_message(session_id, refined_question, full_answer.strip())
        # 6. 키워드 추출 및 저장
        extracted_keyword = await self.openai_repo.extract_keyword(refined_question)
        self.context_repo.save_keywords(extracted_keyword)

    async def load_and_store_pkl(self, file_path: str = "final_result.pkl"):
        import os, pickle
        from app.utils.text_cleaning import clean_text
        """
        .pkl 파일에서 데이터를 로드하고 Milvus에 저장합니다.
        :param file_path: .pkl 파일 경로
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} 파일이 존재하지 않습니다.")

        # .pkl 파일 로드
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # Milvus 초기화
        self.milvus_repo.initialize()
        # self.milvus_repo.delete_all()

        # 데이터 삽입
        for question, answer in data.items():
            # 텍스트 클리닝
            cleaned_question = clean_text(question)
            cleaned_answer = clean_text(answer)

            # 중복 확인
            if self.milvus_repo.is_question_exists(cleaned_question):
                print(f"⚠️ 이미 존재하는 질문: {cleaned_question}")
                continue

            # 질문을 임베딩으로 변환
            embedding = await self.openai_repo.generate_embedding(cleaned_question)

            # Milvus에 데이터 삽입
            self.milvus_repo.insert_faq(cleaned_question, cleaned_answer, embedding)
            print(f"✅ 질문과 응답 저장 완료: {cleaned_question} -> {cleaned_answer}")
    def is_initialized(self):
        """
        Milvus 컬렉션이 초기화되었는지 확인합니다.
        :return: 초기화 여부 (True/False)
        """
        return not self.milvus_repo.collection.is_empty