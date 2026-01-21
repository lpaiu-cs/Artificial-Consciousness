import json
import os
import math
import time
import threading
from datetime import datetime
from typing import List, Dict, Optional
import requests
import config

# --- [심리학적 이론 구현을 위한 상수 및 설정] ---
# 에빙하우스 망각 곡선 상수 (감쇠 속도)
DECAY_RATE = 0.05 
# ACT-R 활성화 가중치
WEIGHT_RECENCY = 0.5    # 최신성 가중치
WEIGHT_IMPORTANCE = 0.3 # 중요도 가중치 (편도체)
WEIGHT_REPETITION = 0.2 # 반복(리허설) 가중치

class MemoryObject:
    """
    기억의 최소 단위 (Chunk).
    보고서 6.1 '메모리 객체 스키마' 구현.
    단순 텍스트뿐만 아니라 메타데이터(시간, 감정, 중요도, 접근 횟수)를 포함함.
    """
    def __init__(self, content, role, importance=1.0, emotion="neutral"):
        self.content = content
        self.role = role # user or assistant
        self.timestamp = time.time() # 생성 시간 (t0)
        self.last_accessed = time.time() # 마지막 인출 시간 (리허설용)
        self.access_count = 1 # 인출 횟수 (n)
        self.importance = importance # 정서적 현저성 (1~10)
        self.emotion = emotion # 당시의 감정 태그 (OCC 모델 기반)

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        obj = cls(data['content'], data['role'])
        obj.timestamp = data['timestamp']
        obj.last_accessed = data['last_accessed']
        obj.access_count = data['access_count']
        obj.importance = data['importance']
        obj.emotion = data.get('emotion', 'neutral')
        return obj

    def get_activation_score(self):
        """
        보고서 3.1 & 6.2: 기억 인출 점수 계산
        Score = (Recency * Decay) + (Importance) + (Repetition)
        에빙하우스의 망각 곡선과 ACT-R 활성화 모델의 간소화된 결합.
        """
        time_diff_hours = (time.time() - self.last_accessed) / 3600
        
        # 1. 최신성(Recency): 시간이 지날수록 지수적으로 감소 (Memory Decay)
        # 중요도가 높을수록(충격적인 사건) 감쇠가 느려짐 (S값 보정)
        stability = 1 + (self.importance * 0.5) 
        recency_score = math.exp(-DECAY_RATE * time_diff_hours / stability)
        
        # 2. 중요도(Importance): 편도체 자극 (영구 기억화)
        importance_score = self.importance / 10.0  # 0.1 ~ 1.0 정규화

        # 3. 반복(Repetition): 많이 회상될수록 강화 (유지 리허설)
        repetition_score = math.log(1 + self.access_count) # 로그 스케일

        # 최종 가중치 합산
        final_score = (WEIGHT_RECENCY * recency_score) + \
                      (WEIGHT_IMPORTANCE * importance_score) + \
                      (WEIGHT_REPETITION * repetition_score)
        
        return final_score

class AdvancedCognitiveMemory:
    def __init__(self, db_path="cognitive_memory.json"):
        self.db_path = db_path
        # 구조: { user_id: { "profile": {...}, "episodic_logs": [MemoryObject...] } }
        self.long_term_memory = self._load_db()
        
        # 감각 등록기 (Sensory Register) - 2.1.1
        # 메시지를 즉시 저장하지 않고 잠시 머무르는 버퍼
        self.sensory_buffer = [] 

    def _load_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                # JSON -> MemoryObject 변환
                restored = {}
                for uid, data in raw_data.items():
                    restored[uid] = {
                        "profile": data.get("profile", {}),
                        "episodic_logs": [MemoryObject.from_dict(m) for m in data.get("episodic_logs", [])]
                    }
                return restored
            except Exception:
                return {}
        return {}

    def _save_db(self):
        # MemoryObject -> JSON 변환
        serialized = {}
        for uid, data in self.long_term_memory.items():
            serialized[uid] = {
                "profile": data.get("profile", {}),
                "episodic_logs": [m.to_dict() for m in data.get("episodic_logs", [])]
            }
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=False, indent=2)

    # --- [1. 감각 등록 및 주의 필터링 (Sensory & Attention)] ---
    def process_input(self, user_id, user_name, content):
        """
        모든 입력이 바로 기억되는 것이 아니다. 
        보고서 2.1.1: '주의(Attention)'를 기울일 가치가 있는지 판단.
        """
        # 간단한 휴리스틱: 너무 짧거나 의미 없는 말은 기억하지 않음 (노이즈 필터링)
        # 실제 구현에선 LLM에게 'Importance Score'를 요청하는 것이 가장 좋음 (여기선 약식 구현)
        importance = 1.0
        emotion = "neutral"
        
        # (임시 로직) 봇이 언급되거나 감정 단어가 있으면 중요도 상승
        if "잼봇" in content or "야" in content: 
            importance += 2.0
        if any(w in content for w in ["사랑", "슬퍼", "화나", "좋아", "행복"]):
            importance += 3.0
            emotion = "emotional"

        # 중요도가 특정 임계값(예: 2.0) 미만이면 단기 기억에도 안 남기고 폐기할 수도 있음
        # 여기서는 일단 모두 Episodic Log로 넘기되, Score가 낮게 책정됨
        
        self._add_episodic_memory(user_id, content, "user", importance, emotion)

    def _add_episodic_memory(self, user_id, content, role, importance, emotion):
        uid = str(user_id)
        if uid not in self.long_term_memory:
            self.long_term_memory[uid] = {"profile": {}, "episodic_logs": []}
        
        # 2.2.1 일화적 기억 생성
        new_memory = MemoryObject(content, role, importance, emotion)
        self.long_term_memory[uid]["episodic_logs"].append(new_memory)
        self._save_db()

    # --- [2. 작업 기억 인출 (Working Memory Retrieval)] ---
    def retrieve_relevant_memories(self, user_id, current_query, limit=5):
        """
        보고서 6.2: 검색 점수 알고리즘에 따른 기억 인출.
        단순 최신순이 아니라, '활성화 수준(Activation Score)'이 높은 기억을 가져옴.
        """
        uid = str(user_id)
        if uid not in self.long_term_memory:
            return [], {}

        memories = self.long_term_memory[uid]["episodic_logs"]
        profile = self.long_term_memory[uid]["profile"]

        # 각 기억의 활성화 점수 계산
        scored_memories = []
        for mem in memories:
            # 텍스트 유사도(Relevance) 로직은 벡터 DB 없이 구현이 무거우므로,
            # 여기서는 Keyword Matching으로 가산점을 주는 방식(Hybrid) 사용
            relevance_bonus = 0
            if any(word in mem.content for word in current_query.split()):
                relevance_bonus = 2.0
            
            score = mem.get_activation_score() + relevance_bonus
            scored_memories.append((score, mem))

        # 점수 높은 순 정렬
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        
        # 상위 K개 추출 (작업 기억으로 전이)
        top_k = [m for score, m in scored_memories[:limit]]
        
        # **리허설 효과(Rehearsal)**: 인출된 기억은 access_count 증가 & 시간 갱신
        for m in top_k:
            m.access_count += 1
            m.last_accessed = time.time() # Refresh
        
        self._save_db() # 갱신된 메타데이터 저장
        return top_k, profile

    # --- [3. 회고 및 공고화 (Reflection & Consolidation)] ---
    def reflect_and_consolidate(self, user_id):
        """
        보고서 6.3: 성찰과 통찰의 생성.
        파편화된 일화적 기억(Episodic)을 의미적 기억(Semantic Profile)으로 변환.
        주기적으로(Sleep Mode) 실행됨.
        """
        uid = str(user_id)
        if uid not in self.long_term_memory: return

        memories = self.long_term_memory[uid]["episodic_logs"]
        # 최근 처리되지 않은 기억들만 모아서 LLM에게 요약 요청
        # (실제 구현 시엔 'processed' 플래그 관리 필요)
        recent_logs = [m.content for m in memories[-10:]] 
        
        if not recent_logs: return

        def _llm_reflection_task():
            try:
                log_text = "\n".join(recent_logs)
                prompt = f"""
                너는 인지 심리학자다. 아래 대화 파편들을 분석하여 유저에 대한 '통찰(Insight)'을 도출하라.
                
                [입력 로그]
                {log_text}
                
                [지시사항]
                1. 단순한 사실 나열이 아니라, 유저의 성격, 현재 고민, 취향을 관통하는 키워드를 뽑아라.
                2. 이 정보는 유저의 '프로필(Semantic Memory)'에 저장된다.
                3. JSON 형식: {{"job": "...", "personality": "...", "current_concern": "..."}}
                """
                
                # LLM 호출 (config.LLM_API_KEY 사용)
                payload = {
                    "model": config.LLM_MODEL,
                    "messages": [{"role": "system", "content": prompt}],
                    "response_format": {"type": "json_object"}
                }
                headers = {"Authorization": f"Bearer {config.LLM_API_KEY}", "Content-Type": "application/json"}
                r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                
                if r.status_code == 200:
                    data = json.loads(r.json()['choices'][0]['message']['content'])
                    
                    # 의미적 기억(Profile) 업데이트 (공고화)
                    profile = self.long_term_memory[uid]["profile"]
                    profile.update(data)
                    self._save_db()
                    print(f"🧠 [Reflection] 유저({uid})의 의미적 기억이 강화되었습니다.")

            except Exception as e:
                print(f"Reflection Error: {e}")

        threading.Thread(target=_llm_reflection_task).start()

    # --- [LLM Handler용 Context String 생성기] ---
    def get_context_string(self, user_id, current_query):
        """
        보고서 5.1: 기억의 재구성을 위한 Raw Data 제공.
        LLM에게 이 데이터를 주고 '재구성(Reconstruction)'하도록 유도해야 함.
        """
        top_memories, profile = self.retrieve_relevant_memories(user_id, current_query)
        
        # 의미적 기억 (Semantic)
        semantic_str = ", ".join([f"{k}: {v}" for k,v in profile.items()])
        
        # 일화적 기억 (Episodic) - 시간 정보 포함
        episodic_str = ""
        for m in top_memories:
            # 타임스탬프를 '3일 전', '방금' 등으로 변환하는 로직이 있으면 더 좋음
            date_str = datetime.fromtimestamp(m.timestamp).strftime("%Y-%m-%d %H:%M")
            episodic_str += f"- [{date_str}] (중요도 {m.importance}): {m.content}\n"

        return f"""
        [User Profile (Semantic Memory)]
        {semantic_str}
        
        [Related Memories (Episodic Memory)]
        {episodic_str}
        (위 기억들은 현재 상황과 가장 관련성이 높거나, 강렬하게 남아있는 기억들입니다.)
        """