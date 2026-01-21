import requests
import re  # 정규식 모듈
import config
from memory_handler import MemoryManager
import threading # 회고 기능을 비동기로 돌리기 위해 (선택사항)

BOT_PERSONA = config.BOT_PERSONA

class LLMHandler:
    def __init__(self):
        self.api_key = config.LLM_API_KEY
        self.url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.memory_manager = MemoryManager()
        self.turn_count = 0 # 회고 주기 체크용

    def get_response(self, history, current_user_query=None):
        self.turn_count += 1
        
        # 0. 들어온 메시지 즉시 스타일 분석 (Regex - Fast)
        if history:
            last_msg = history[-1]
            if str(last_msg.get('user_id')) != str(config.BOT_USER_ID):
                self.memory.analyze_and_update_style(last_msg.get('user_id'), last_msg.get('msg', ''))

        # 1. 프롬프트 구성을 위한 데이터 준비
        # 현재 대화에 참여 중인 유저들의 ID 추출 (최근 10개 메시지 기준)
        active_users = list(set([str(h.get('user_id')) for h in history[-10:] if str(h.get('user_id')) != str(config.BOT_USER_ID)]))
        
        # 기억(Memory) + 스타일(Style) 로딩
        memory_context = self.memory.get_user_context(active_users)
        
        # In-Context Learning을 위한 실제 대화 예시 (최근 3개만, 토큰 절약)
        recent_examples = "\n".join([f"{h.get('user_name', 'User')}: {h.get('msg')}" for h in history[-3:]])

        # 2. 시스템 프롬프트 조립 (페르소나 + 기억 + 지침)
        system_prompt = f"""
{BOT_PERSONA}

[현재 대화 참여자 정보 (기억 데이터)]
{memory_context}

[행동 지침]
1. 위 '기억 데이터'를 참고하여 아는 척을 해라.
2. 사용자의 말투 특징(반말/존댓말 등)이 정보에 있다면 그에 맞춰 '미러링(Mirroring)' 해라.
3. 현재 대화 흐름상 네가 굳이 끼어들 필요가 없거나, 할 말이 없으면 고민하지 말고 
   반드시 `[PASS]` 라고만 출력해라. (매우 중요: 눈치 없이 아무 때나 끼어들지 말 것)
4. 답변은 JSON이 아닌 자연스러운 텍스트로 해라.

[최근 대화 분위기 참고]
{recent_examples}
"""

        input_messages = [{"role": "system", "content": system_prompt}]
        
        # 히스토리 주입 (토큰 효율을 위해 최근 15개)
        for h in history[-15:]:
            role = "assistant" if str(h.get("user_id")) == str(config.BOT_USER_ID) else "user"
            prefix = f"{h.get('user_name', 'User')}: " if role == "user" else ""
            input_messages.append({"role": role, "content": f"{prefix}{h.get('msg')}"})

        if current_user_query:
            input_messages.append({"role": "user", "content": current_user_query})

        # 3. LLM API 호출
        payload = {
            "model": config.LLM_MODEL,
            "messages": input_messages,
            "temperature": 0.8, # 페르소나 연기를 위해 약간 창의적으로
            "max_tokens": 800
        }

        try:
            # 5턴마다 비동기 회고 실행 (기억 업데이트)
            if self.turn_count % 5 == 0:
                self.memory.reflect_async(history[-10:])

            r = requests.post(self.url, headers=self.headers, json=payload, timeout=10)
            if r.status_code == 200:
                content = r.json()['choices'][0]['message']['content'].strip()
                
                # [PASS] 토큰 감지 (Regex) - 대소문자/괄호 변형 대응
                if re.search(r"\[PASS\]", content, re.IGNORECASE):
                    print(f"🤫 봇 침묵 (Reason: {content})")
                    return None
                
                return content
            
            else:
                return f"🤖 (오류: {r.status_code})"

        except Exception as e:
            return f"🤖 (연결 실패: {e})"