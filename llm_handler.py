import threading
import requests
import config
import re
from memory_structures import ImportanceScorer
from memory_handler import FastSTM, SlowLTM

class LLMHandler:
    def __init__(self):
        self.scorer = ImportanceScorer(bot_name="잼봇")
        self.stm = FastSTM(capacity=15) # 단기 기억 용량
        self.ltm = SlowLTM()            # 장기 기억 관리자
        
        self.headers = {
            "Authorization": f"Bearer {config.LLM_API_KEY}",
            "Content-Type": "application/json",
        }

    def get_response(self, history, current_query=None):
        # 1. [System 1] 입력된 메시지 중요도 채점 및 STM 저장
        if history:
            last_msg = history[-1]
            uid = str(last_msg.get('user_id'))
            msg = last_msg.get('msg', '')
            role = "assistant" if uid == str(config.BOT_USER_ID) else "user"
            
            # ⚡ NLP 기반 즉시 채점 (LLM 호출 X)
            importance = self.scorer.score(msg, role)
            
            # STM에 저장 (용량 차면 자동으로 Buffer로 밀려남)
            self.stm.add(msg, uid, role, importance)

        # 2. [Context Building] 프롬프트 구성
        # 2-1. STM에서 살아남은 '중요한 단기 기억' 가져오기
        active_stm = self.stm.get_working_context()
        stm_context = "\n".join([f"{m.user_id}: {m.content}" for m in active_stm])
        
        # 2-2. 현재 참여자의 LTM(장기 기억) 가져오기
        active_users = list(set([m.user_id for m in active_stm if m.role == 'user']))
        ltm_context = self.ltm.get_context_string(active_users)

        system_prompt = f"""
{config.BOT_PERSONA}

[Long-term Memory: 유저 정보]
{ltm_context}

[Working Memory: 현재 대화 흐름]
(오래된 잡담은 사라지고 중요한 맥락만 남아있다)
{stm_context}

[Instruction]
1. 위 기억을 바탕으로 자연스럽게 대화해라.
2. 할 말이 없거나 끼어들 타이밍이 아니면 `[PASS]` 라고만 출력해라.
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
        if len(self.stm.transfer_buffer) >= 5:
            buffer_snapshot = self.stm.transfer_buffer[:] # 복사
            self.stm.transfer_buffer = [] # 버퍼 비우기
            
            # 비동기 실행
            t = threading.Thread(target=self.ltm.consolidate, args=(buffer_snapshot,))
            t.start()