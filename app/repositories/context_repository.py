from typing import Dict, List

class ContextRepository:
    # 슬라이딩 윈도우 크기 설정
    WINDOW_SIZE = 3

    def __init__(self):
        # 세션 데이터를 메모리에 저장 (필요시 Redis 등으로 대체 가능)
        self.session_data: Dict[str, List[Dict[str, str]]] = {}

    def create_session(self) -> str:
        """
        새로운 세션 ID를 생성합니다.
        :return: 생성된 세션 ID
        """
        import uuid
        session_id = str(uuid.uuid4())
        self.session_data[session_id] = []
        return session_id

    def get_context(self, session_id: str) -> str:
        """
        세션 ID를 기반으로 이전 대화 맥락을 가져옵니다.
        :param session_id: 사용자 세션 ID
        :return: 이전 대화 맥락 (문자열)
        """
        if session_id not in self.session_data:
            return ""
        return "\n".join([
            f"User: {msg['question']}\nBot: {msg['response']}"
            for msg in self.session_data[session_id]
        ])

    def save_user_message(self, session_id: str, question: str, response: str):
        """
        사용자 질문과 봇 응답을 세션 데이터에 저장합니다.
        :param session_id: 사용자 세션 ID
        :param question: 사용자가 입력한 질문
        :param response: 봇의 응답
        """
        if session_id not in self.session_data:
            self.session_data[session_id] = []
        self.session_data[session_id].append({"question": question, "response": response})

        # 슬라이딩 윈도우 방식으로 최대 WINDOW_SIZE개의 대화만 유지
        if len(self.session_data[session_id]) > self.WINDOW_SIZE:
            self.session_data[session_id] = self.session_data[session_id][-self.WINDOW_SIZE:]

    def get_important_context(self, session_id: str) -> str:
        """
        중요도가 높은 최근 대화 맥락을 가져옵니다.
        :param session_id: 사용자 세션 ID
        :return: 최근 대화 맥락 (문자열)
        """
        if session_id not in self.session_data:
            return ""
        important_messages = self.session_data[session_id][-self.WINDOW_SIZE:]  # 최근 WINDOW_SIZE개의 대화만 포함
        return "\n".join([
            f"User: {msg['question']}\nBot: {msg['response']}"
            for msg in important_messages
        ])