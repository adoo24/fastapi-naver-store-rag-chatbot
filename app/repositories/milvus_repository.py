from pymilvus import CollectionSchema, DataType, FieldSchema, connections, Collection, utility
import json

class MilvusRepository:
    def __init__(self):
        self.collection_name = "faq_collection"
        self.collection = None
        self.initialize()

    def initialize(self):
        """
        Milvus 초기화
        """
        connections.connect("default", host="localhost", port="19530")
        if not utility.has_collection(self.collection_name):
            self.init_milvus()
        self.collection = Collection(self.collection_name)
        self.collection.load()

    def is_question_exists(self, question: str) -> bool:
        """
        질문이 Milvus에 존재하는지 확인
        """
        # 질문 문자열을 JSON 형식으로 이스케이프 처리
        escaped_question = json.dumps(question)

        search_results = self.collection.query(
            expr=f"question == {escaped_question}",
            output_fields=["question"],
            limit=1
        )
        return len(search_results) > 0
    

    def find_similar_faqs(self, embedding, top_k: int = 10):
        """
        Milvus에서 유사 질문 검색
        """
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["question", "answer"]
        )
        return [
            {"question": hit.entity.get("question"), "answer": hit.entity.get("answer")}
            for hits in results
            for hit in hits if hit.score > 0.5
        ]
    def delete_all(self):
        """
        Milvus에서 모든 데이터 삭제
        """
        self.collection.delete(expr="question != ''")

    def insert_faq(self, cleaned_question, cleaned_answer, embedding):
        """
        Milvus에 데이터 삽입
        """
        if self.is_question_exists(cleaned_question):
            print(f"⚠️ 이미 존재하는 질문: {cleaned_question}")
            return

        self.collection.insert([[cleaned_question], [cleaned_answer], [embedding]])
        print(f"✅ 질문과 응답 저장 완료: {cleaned_question} -> {cleaned_answer}")
    def init_milvus(self):
        connections.connect("default", host="localhost", port="19530")
        collection_name = "faq_collection"
        
        # 새 컬렉션 생성
        fields = [
            FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=500, is_primary=True),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=65535),  # 응답 필드 추가
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)
        ]
        schema = CollectionSchema(fields=fields, description="FAQ Collection")
        
        collection = Collection(collection_name, schema=schema, using="default")
        
        # 인덱스 생성
        if not collection.has_index():
            index_params = {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}}
            collection.create_index(field_name="embedding", index_params=index_params)
            print("✅ 인덱스 생성 완료")
        
        collection.load()
        return collection