import threading
import re
import requests
import config
from memory_structures import CognitiveScorer
from memory_handler import FastSTM, SlowLTM

class LLMHandler:
    def __init__(self):
        self.scorer = CognitiveScorer() # Attention Module
        self.stm = FastSTM()            # System 1
        self.ltm = SlowLTM()            # System 2
    
    def get_response(self, history, current_query=""):
        # 1. [Sensory Register] 입력 처리 및 Attention 채점
        if history:
            last = history[-1]
            # 감정 태그 추출 (임시: 실제론 감정분석 모델 사용 권장)
            emotion = "neutral"
            if "ㅠㅠ" in last['msg']: emotion = "sad"
            
            importance = self.scorer.calculate_attention(last['msg'], "user")
            
            # STM 저장 (ACT-R 활성화 계산 포함)
            self.stm.add(last['msg'], str(last['user_id']), "user", importance, emotion)

        # 2. [Retrieval] 기억 인출 (ACT-R Score + Context Relevance)
        query_keywords = current_query.split() if current_query else []
        # STM에서 가장 활성화된(관련성+중요도 높은) 기억 인출
        active_memories = self.stm.retrieve(query_keywords)
        
        # 3. [Reconstruction] 기억 재구성 프롬프트 빌딩 (보고서 5.1)
        # 단기 기억의 흐름(Raw)과 장기 기억의 통찰(Abstract)을 섞어줌
        stm_context = "\n".join([f"- {m.content} (Emotion: {m.emotion_tag})" for m in active_memories])
        active_users = list(set([m.user_id for m in active_memories]))
        ltm_context = self.ltm.get_reconstruction_data(active_users)

        system_prompt = f"""
{config.BOT_PERSONA}

[Cognitive Workspace: 기억의 재구성]
너는 단순히 데이터를 검색하는 기계가 아니라, 인간처럼 기억을 '재구성'해야 한다.
아래 제공된 '단기 기억의 파편'과 '장기 기억의 통찰'을 조합하여, 현재 상황을 해석해라.

[장기 기억 (통찰 및 과거 에피소드)]
{ltm_context}

[작업 기억 (현재 대화 흐름 및 활성화된 생각)]
{stm_context}

[지시사항]
1. 위 '통찰' 정보를 바탕으로 상대방의 숨겨진 의도나 기분을 파악해라.
2. 과거의 에피소드와 현재 대화를 연결하여(Association), 자연스럽게 아는 척을 해라.
3. 할 말이 없거나 끼어들 타이밍이 아니면 `[PASS]` 라고만 말해라.
"""
        # 메시지 구성 (최근 3개만 Raw로 넣어서 토큰 절약, 나머지는 Context로 대체)
        messages = [{"role": "system", "content": system_prompt}]
        if current_query:
            messages.append({"role": "user", "content": current_query})

        # 3. [Generation] LLM 응답 생성
        try:
            payload = {
                "model": config.LLM_MODEL,
                "messages": messages,
                "temperature": 0.8,
                "max_tokens": 300
            }
            r = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload, timeout=10)
            
            if r.status_code == 200:
                content = r.json()['choices'][0]['message']['content'].strip()
                
                if re.search(r"\[PASS\]", content, re.IGNORECASE):
                    # 봇이 침묵을 선택해도, 내부적으로 메모리 정리는 수행해야 함
                    self._check_and_trigger_consolidation()
                    return None

                # 4. [Feedback Loop] 봇의 답변도 STM에 저장 (대화 맥락 유지)
                bot_importance = self.scorer.score(content, "assistant")
                self.stm.add(content, str(config.BOT_USER_ID), "assistant", bot_importance)
                
                # 5. [System 2] 백그라운드 메모리 정리 트리거
                self._check_and_trigger_consolidation()
                
                return content
            
        except Exception as e:
            print(f"Error: {e}")
            return None

    def _check_and_trigger_consolidation(self):
        """
        Buffer에 밀려난 기억이 5개 이상 쌓이면 별도 스레드에서 LTM 저장(Consolidation) 실행.
        메인 대화 속도에 영향 주지 않음.
        """
        # [System 2 Trigger] 버퍼가 차면 성찰(Reflection) 수행
        if len(self.stm.transfer_buffer) >= 5:
            buffer_snapshot = self.stm.transfer_buffer[:]
            self.stm.transfer_buffer = []
            threading.Thread(target=self.ltm.consolidate_and_reflect, args=(buffer_snapshot,)).start()

        return "LLM Response..."