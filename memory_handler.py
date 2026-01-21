import heapq
import json
import os
import requests
import config
from memory_structures import MemoryObject, ImportanceScorer

class FastSTM:
    """
    [단기 기억] 작업 기억(Working Memory).
    우선순위 큐(Min-Heap)를 사용하여 용량 초과 시 '가치 낮은' 기억부터 방출.
    """
    def __init__(self, capacity=15):
        self.capacity = capacity
        self.memory_queue = [] # Heap
        self.transfer_buffer = [] # LTM으로 보낼 방출된 기억들 (Overflow)

    def add(self, content, user_id, role, importance):
        mem = MemoryObject(content, user_id, role, importance)
        heapq.heappush(self.memory_queue, mem)
        
        # [cite_start]용량 관리 (Eviction Process) [cite: 5]
        if len(self.memory_queue) > self.capacity:
            # 점수가 가장 낮은 기억(오래됐거나 중요하지 않음)이 pop됨
            evicted = heapq.heappop(self.memory_queue)
            
            # 방출된 기억 중, 중요도가 2.0 이상인 것만 LTM 후보로 보냄 (잡담은 영구 삭제)
            if evicted.importance >= 2.0:
                self.transfer_buffer.append(evicted)
                # print(f"👋 [STM -> Buffer] '{evicted.content}' (Score: {evicted.importance})")

    def get_working_context(self):
        """프롬프트에 주입할 현재 활성 기억들 (시간순 정렬)"""
        # 힙을 깨지 않고 리스트로 변환 후, 타임스탬프 순 정렬
        sorted_mems = sorted(self.memory_queue, key=lambda x: x.timestamp)
        return [m for m in sorted_mems]

class SlowLTM:
    """
    [장기 기억] 의미 기억(Profile) 및 일화 기억(Episode).
    Buffer에 쌓인 기억을 LLM으로 '회고(Consolidation)'하여 저장.
    """
    def __init__(self, db_path="ltm_storage.json"):
        self.db_path = db_path
        # { user_id: { "profile": {...}, "episodes": [...] } }
        self.data = self._load_db()

    def _load_db(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_db(self):
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def consolidate(self, buffer_items):
        """
        [cite_start][System 2] 방출된 기억들을 모아 LLM에게 분석 요청. [cite: 51, 53]
        """
        if not buffer_items: return

        # 분석용 텍스트 변환
        log_text = "\n".join([f"{m.user_id}: {m.content}" for m in buffer_items])
        
        # LLM 호출: 사실(Fact)과 에피소드(Episode) 추출
        try:
            prompt = f"""
            너는 기억 관리자다. 아래는 단기 기억에서 밀려난 대화 조각들이다.
            여기서 유저에 대한 불변의 '사실(Fact)'이나 기억해둘 만한 '사건(Episode)'만 JSON으로 추출해라.
            없으면 빈 리스트를 반환해.
            
            [Input]
            {log_text}
            
            [Output Format JSON]
            {{
                "user_id": {{
                    "facts": {{ "취미": "독서" }},
                    "episodes": ["2026-01-21에 야근 때문에 힘들어함"]
                }}
            }}
            """
            
            headers = {"Authorization": f"Bearer {config.LLM_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": config.LLM_MODEL,
                "messages": [{"role": "system", "content": prompt}],
                "response_format": {"type": "json_object"}
            }
            
            r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=10)
            if r.status_code == 200:
                result = json.loads(r.json()['choices'][0]['message']['content'])
                self._update_storage(result)
                # print(f"🧠 [Consolidation] 장기 기억 업데이트 완료 ({len(buffer_items)}개 항목 처리)")

        except Exception as e:
            print(f"LTM Consolidation Error: {e}")

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

    def get_context_string(self, user_ids):
        """특정 유저들의 장기 기억(프로필+최근 에피소드) 반환"""
        lines = []
        for uid in user_ids:
            if uid in self.data:
                p = self.data[uid]["profile"]
                e = self.data[uid]["episodes"][-3:] # 최근 3개만
                p_str = ", ".join([f"{k}:{v}" for k,v in p.items()])
                e_str = " / ".join([x['summary'] for x in e])
                lines.append(f"- User({uid}): [특징] {p_str} [최근일화] {e_str}")
        return "\n".join(lines)