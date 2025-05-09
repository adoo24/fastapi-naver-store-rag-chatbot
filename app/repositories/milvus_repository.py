from pymilvus import connections, Collection, utility

class MilvusRepository:
    def __init__(self):
        self.collection_name = "faq_collection"
        self.collection = None

    def initialize(self):
        """
        Milvus 초기화
        """
        connections.connect("default", host="localhost", port="19530")
        if not utility.has_collection(self.collection_name):
            raise ValueError(f"컬렉션 '{self.collection_name}'이 존재하지 않습니다.")
        self.collection = Collection(self.collection_name)
        self.collection.load()

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
            for hit in hits
        ]