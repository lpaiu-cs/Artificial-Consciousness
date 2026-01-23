import json
import threading
import requests
import config
from memory_structures import CognitiveScorer, MemoryObject
from memory_handler import FastSTM, SlowLTM
from social_emotion_handler import BotEmotionEngine, SocialMemoryEngine
from vector_engine import VectorEngine

class LLMHandler:
    def __init__(self):
        # 1. 인지 모듈 (감각 & 의미)
        self.scorer = CognitiveScorer()
        self.vector_engine = VectorEngine()
        
        # 2. 기억 모듈 (System 1 & 2)
        self.stm = FastSTM(capacity=15) # 단기 기억 용량
        self.ltm = SlowLTM()            # 장기 기억 관리자
        
        # 3. 정서/사회 모듈 (마음 & 관계)
        self.emotion_engine = BotEmotionEngine()
        self.social_engine = SocialMemoryEngine()

        self.headers = {
            "Authorization": f"Bearer {config.LLM_API_KEY}",
            "Content-Type": "application/json",
        }

    def get_response(self, history, current_msg_data=None):

        if not current_msg_data:
            return None
        
        # --- [Step 1: 지각 (Sensory & Encoding)] ---
        uid = str(current_msg_data.get('user_id'))
        name = current_msg_data.get('user_name', 'Unknown')
        msg = current_msg_data.get('msg', '')
        role = "user" if uid != str(config.BOT_USER_ID) else "assistant"

        if role == "user":
            # 감정/사회성/중요도 업데이트
            self.emotion_engine.update_mood(msg)
            self.social_engine.update_affinity(uid, msg)
            importance = self.scorer.calculate_attention(msg, role)
            
            # 임베딩 생성 (유저 메시지만)
            vector = self.vector_engine.get_embedding(msg)
            
            # STM 저장 (여기서 용량 초과 시 Buffer로 밀려남)
            self.stm.add(msg, uid, role, importance, "neutral", vector)

        # --- [Step 2: 기억 인출 (Hybrid Retrieval)] ---
        query_vec = self.vector_engine.get_embedding(msg) if msg else None
        
        # 의미(Vector) + 맥락(ACT-R) 하이브리드 검색
        active_memories = self.stm.retrieve_hybrid(query_vec, self.vector_engine)
        stm_context = "\n".join([f"- {m.content}" for m in active_memories])
        



        
        # LTM 통찰 및 상태 정보 가져오기
        target_uid = str(current_msg_data.get('user_id')) if current_msg_data else "unknown"
        ltm_context = self.ltm.get_reconstruction_data([target_uid])
        mood_desc = self.emotion_engine.get_mood_description()
        rel_desc = self.social_engine.get_relationship_context(target_uid)

        # --- [Step 3: 메타 인지 (Think & Plan)] ---
        system_prompt = f"""
{config.BOT_PERSONA}

[Internal State]
- Mood: {mood_desc}
- Relationship: {rel_desc}

[Memory Context]
- Long-term Insights: {ltm_context}
- Working Memory: {stm_context}

[Instruction]
JSON 포맷으로 사고 과정(Think)과 행동(Act)을 출력해라.
1. **analysis**: 의도 파악.
2. **strategy**: 감정과 관계에 따른 태도 결정.
3. **decision**: "SPEAK" or "PASS" (낄끼빠빠 판단).
4. **response**: 답변 내용.
"""
        
        final_response = None

        try:
            payload = {
                "model": config.LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": msg or "..."} 
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.8 # 최신 모델에는 온도 조절값이 없음 주의.
            }
            
            r = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload, timeout=15)
            if r.status_code == 200:
                result = json.loads(r.json()['choices'][0]['message']['content'])
                decision = result.get("decision", "PASS")
                response_text = result.get("response", "")

                if decision == "SPEAK" and response_text:
                    final_response = response_text
                    
                    # 봇의 발화도 기억에 저장 (맥락 유지)
                    # 봇의 말은 중요도(2.0)를 주어 STM에 당분간 남게 함
                    self.stm.add(final_response, config.BOT_USER_ID, "assistant", 2.0, "neutral", self.vector_engine.get_embedding(final_response))
                    self.emotion_engine.decay_mood() # 말하고 나면 감정 식힘

        except Exception as e:
            print(f"Handler Error: {e}")

        # --- [Step 4: 기억 공고화 트리거 (Background Consolidation)] ---
        # 중요: 봇이 말을 했든(SPEAK) 안 했든(PASS), 
        # STM 버퍼에 밀려난 기억이 쌓여있으면 정리를 시작해야 함.
        self._check_and_trigger_consolidation()

        return final_response

    def _check_and_trigger_consolidation(self):
        """
        [System 2 Trigger]
        단기 기억에서 밀려난(Evicted) 기억들이 버퍼에 5개 이상 쌓이면,
        별도 스레드에서 LTM(SlowLTM)에게 '성찰 및 저장'을 요청함.
        """
        if len(self.stm.transfer_buffer) >= 5:
            # 버퍼를 비우고 스냅샷을 뜸 (Thread-safe를 위해 복사)
            buffer_snapshot = self.stm.transfer_buffer[:]
            self.stm.transfer_buffer = [] 
            
            print(f"🔄 [Background] 기억 정리 시작 ({len(buffer_snapshot)}개 항목)...")
            
            # 비동기 실행 (메인 대화 흐름 막지 않음)
            t = threading.Thread(target=self.ltm.consolidate_and_reflect, args=(buffer_snapshot,))
            t.start()