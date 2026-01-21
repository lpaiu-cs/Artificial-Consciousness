import numpy as np
import requests
import config

class VectorEngine:
    """
    [Feature 3: Semantic Brain]
    텍스트를 벡터로 변환하고 코사인 유사도를 계산합니다.
    """
    def __init__(self):
        self.api_key = config.LLM_API_KEY
        # 비용 효율적인 모델 사용 (text-embedding-3-small)
        self.model = "text-embedding-3-small"

    def get_embedding(self, text):
        """텍스트 -> 1536차원 벡터 변환"""
        if not text: return None
        try:
            url = "https://api.openai.com/v1/embeddings"
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            data = {"input": text, "model": self.model}
            
            response = requests.post(url, headers=headers, json=data, timeout=5)
            if response.status_code == 200:
                return response.json()['data'][0]['embedding']
        except Exception as e:
            print(f"Embedding Error: {e}")
        return None

    def cosine_similarity(self, vec_a, vec_b):
        """두 벡터 간의 유사도 계산 (-1.0 ~ 1.0)"""
        if vec_a is None or vec_b is None: return 0.0
        
        a = np.array(vec_a)
        b = np.array(vec_b)
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0: return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    def calculate_hybrid_score(self, memory_obj, query_vec):
        """
        [Hybrid Search Formula]
        기억 인출 점수 = (ACT-R 활성화 * 0.4) + (의미적 유사도 * 0.6)
        """
        # 1. 의미적 유사도 (Semantic)
        semantic_score = 0.0
        if query_vec and memory_obj.embedding:
            semantic_score = self.cosine_similarity(query_vec, memory_obj.embedding)
            # 유사도가 0.3 미만이면 노이즈로 간주, 0 처리
            if semantic_score < 0.3: semantic_score = 0.0

        # 2. ACT-R 기저 활성화 (Temporal + Importance)
        # get_base_activation 결과는 보통 -2 ~ +5 사이 값이므로 정규화 필요
        act_score = memory_obj.get_base_activation()
        
        # 간단한 가중치 합산 (상황에 따라 튜닝 필요)
        final_score = (act_score * 0.4) + (semantic_score * 0.6 * 5.0) 
        # *5.0은 코사인 유사도(0~1)를 ACT-R 스케일에 맞추기 위함
        
        return final_score