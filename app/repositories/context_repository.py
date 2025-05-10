from typing import Dict, List
import redis
import json

class ContextRepository:
    # 슬라이딩 윈도우 크기 설정
    WINDOW_SIZE = 3

    def __init__(self, redis_host="localhost", redis_port=6379, db=0):
        # Redis 클라이언트 초기화
        self.client = redis.StrictRedis(host=redis_host, port=redis_port, db=db, decode_responses=True)

    def create_session(self) -> str:
        """
        새로운 세션 ID를 생성합니다.
        :return: 생성된 세션 ID
        """
        session_id = self.client.incr("session_id_counter")
        return f"session_{session_id}"

    def get_context(self, session_id: str) -> str:
        """
        세션 ID를 기반으로 이전 대화 맥락을 가져옵니다.
        :param session_id: 사용자 세션 ID
        :return: 이전 대화 맥락 (문자열)
        """
        context_key = f"context:{session_id}"
        messages = self.client.lrange(context_key, 0, -1)
        return "\n".join([json.loads(msg)["question"] for msg in messages])

    def save_user_message(self, session_id: str, question: str, response: str):
        """
        사용자 질문과 봇 응답을 세션 데이터에 저장합니다.
        :param session_id: 사용자 세션 ID
        :param question: 사용자가 입력한 질문
        :param response: 봇의 응답
        """
        context_key = f"context:{session_id}"
        message = {"question": question, "response": response}
        self.client.rpush(context_key, json.dumps(message))

        # 슬라이딩 윈도우 방식으로 최대 WINDOW_SIZE개의 대화만 유지
        if self.client.llen(context_key) > self.WINDOW_SIZE:
            self.client.ltrim(context_key, -self.WINDOW_SIZE, -1)

    def get_important_context(self, session_id: str) -> str:
        """
        중요도가 높은 최근 대화 맥락을 가져옵니다.
        :param session_id: 사용자 세션 ID
        :return: 최근 대화 맥락 (문자열)
        """
        context_key = f"context:{session_id}"
        messages = self.client.lrange(context_key, -self.WINDOW_SIZE, -1)
        return "\n".join([json.loads(msg)["question"] for msg in messages])
    
    def save_keywords(self, keywords: List[str]):
        """
        키워드 출현 빈도를 업데이트합니다.
        :param keywords: 키워드 리스트
        """
        keyword_key = "keywords"
        for keyword in keywords:
            self.client.hincrby(keyword_key, keyword, 1)
    
    def log_insuffiecient_context_question(self, question: str, embedding: List[float]):
        """
        유사 질문이 부족한 경우 질문과 임베딩을 로그로 저장합니다.
        hash 구조를 이용하여 질문과 임베딩을 저장합니다.
        :param question: 질문
        :param embedding: 질문의 임베딩
        """
        log_key = "insufficient_context_questions"
        question_key = f"{question}"
        self.client.hset(log_key, question_key, json.dumps(embedding))