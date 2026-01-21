import heapq
import json
import time
import threading
import requests
import config
from memory_structures import MemoryObject, CognitiveScorer, NOISE_THRESHOLD

class FastSTM:
    """
    [System 1] 작업 기억 (Working Memory)
    보고서 2.1.2: 단순 저장이 아닌, '중앙 집행기'에 의한 관리.
    우선순위 큐를 ACT-R 활성화 점수 기반으로 운영.
    """
    def __init__(self, capacity=15):
        self.capacity = capacity
        self.memory_queue = [] # Min-Heap (Base Activation 기준 정렬)
        self.transfer_buffer = [] # LTM 전송용 버퍼

    def add(self, content, user_id, role, importance, emotion="neutral"):
        # [Attention Filter] 중요도가 너무 낮으면 아예 기억하지 않음 (보고서 2.1.1)
        if importance < NOISE_THRESHOLD:
            return 

        mem = MemoryObject(content, user_id, role, importance, emotion)
        heapq.heappush(self.memory_queue, mem)
        
        # [Eviction] 용량 초과 시 '활성화 수준'이 가장 낮은 기억 방출
        if len(self.memory_queue) > self.capacity:
            evicted = heapq.heappop(self.memory_queue)
            
            # 방출된 기억 중 가치 있는 것만 LTM 후보로 (망각 곡선에 따른 자연 소멸)
            # 중요도 3.0 이상이거나, 최근에 자주 인출되었던 기억은 살림
            if evicted.importance >= 3.0 or len(evicted.access_history) > 2:
                self.transfer_buffer.append(evicted)

    def retrieve(self, query_keywords):
        """
        [보고서 6.2] 검색 점수 (Retrieval Score) 계산
        Final Score = (Base Activation) + (Context Relevance)
        """
        scored_memories = []
        for mem in self.memory_queue:
            # 1. 기저 활성화 (ACT-R: 빈도 + 최신성 + 중요도)
            base_act = mem.get_base_activation()
            
            # 2. 문맥 관련성 (Keyword Overlap - 약식 구현)
            # 실제로는 Vector Embedding Cosine Similarity가 가장 정확함
            relevance = 0
            if any(k in mem.content for k in query_keywords):
                relevance = 2.0  # 관련성 보너스 (W3)
            
            final_score = base_act + relevance
            scored_memories.append((final_score, mem))
        
        # 점수 높은 순 정렬 (인출)
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        
        # [보고서 2.1.3] 리허설 효과: 인출된 기억은 강화됨
        retrieved = [m for score, m in scored_memories]
        for m in retrieved:
            m.access_history.append(time.time()) # 현재 시간 추가 (활성화 상승)
            
        return retrieved

class SlowLTM:
    """
    [System 2] 장기 기억 & 성찰 (Reflection)
    보고서 6.3: 단순 저장이 아닌 '성찰'과 '통찰' 생성.
    """
    def __init__(self, db_path="cognitive_ltm.json"):
        self.db_path = db_path
        self.data = self._load_db()

    def _load_db(self):
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {} # {user_id: {"insights": [], "episodes": []}}

    def consolidate_and_reflect(self, buffer_items):
        """
        [보고서 6.3] 성찰(Reflection) 프로세스
        단기 기억 파편들을 모아 '상위 수준의 통찰(Insight)'을 생성.
        """
        if not buffer_items: return

        log_text = "\n".join([f"{m.user_id}: {m.content}" for m in buffer_items])
        
        # LLM에게 '성찰' 요청
        prompt = f"""
        너는 인지 심리학자다. 아래는 단기 기억에서 넘어온 대화 조각들이다.
        이것들을 분석하여 다음 두 가지 형태로 '기억의 재구성'을 수행해라.

        1. **Episodes (일화)**: "언제, 누가, 무엇을 했는지" 구체적인 사건 요약.
        2. **Insights (통찰)**: 이 대화를 통해 알게 된 유저의 '심층적인 성향'이나 '숨겨진 의도'.
           (단순한 사실 나열이 아니라, 추상화된 지식을 원함. 예: "A는 요즘 진로 문제로 불안해하고 있다.")

        [입력 로그]
        {log_text}

        [출력 포맷 JSON]
        {{
            "user_id": {{
                "episodes": ["날짜 불명 - A와 B가 싸움"],
                "insights": ["A는 경쟁적인 상황을 싫어함"]
            }}
        }}
        """
        
        try:
            headers = {"Authorization": f"Bearer {config.LLM_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": config.LLM_MODEL, "messages": [{"role": "system", "content": prompt}], "response_format": {"type": "json_object"}}
            
            r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=15)
            if r.status_code == 200:
                result = json.loads(r.json()['choices'][0]['message']['content'])
                self._update_storage(result)
        except Exception as e:
            print(f"Reflection Error: {e}")

    def _update_storage(self, extracted_data):
        for uid, info in extracted_data.items():
            if uid not in self.data: self.data[uid] = {"profile": {}, "episodes": []}
            
            # Update Profile
            if "facts" in info:
                self.data[uid]["profile"].update(info["facts"])
            
            # Append Episodes
            if "episodes" in info:
                # 중복 방지 후 추가
                existing = set(e['summary'] for e in self.data[uid]["episodes"])
                for ep in info["episodes"]:
                    if ep not in existing:
                        self.data[uid]["episodes"].append({"date": "Today", "summary": ep})
        
        self.save_db()

def get_reconstruction_data(self, user_ids):
        """
        [보고서 5.1] 기억 재구성을 위한 Raw Data 제공
        LLM이 이 데이터를 보고 '이야기'를 재구성하도록 함.
        """
        context = []
        for uid in user_ids:
            if uid in self.data:
                # 통찰(Insight)이 에피소드보다 더 상위에 옴
                insights = " / ".join(self.data[uid].get("insights", [])[-3:])
                episodes = " / ".join(self.data[uid].get("episodes", [])[-3:])
                context.append(f"User({uid}) - [통찰]: {insights}\n    - [기억]: {episodes}")
        return "\n".join(context)