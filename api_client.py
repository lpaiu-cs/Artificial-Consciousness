import logging
import json
from typing import List, Any, Union

# 외부 라이브러리
from openai import OpenAI
from groq import Groq

# 설정 로드
import config

class UnifiedAPIClient:
    """
    [API Gateway]
    OpenAI와 Groq 클라이언트를 통합 관리합니다.
    """
    def __init__(self):
        # 1. OpenAI Init
        if config.OPENAI_API_KEY:
            self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        else:
            logging.warning("⚠️ OpenAI API Key is missing.")
            self.openai_client = None

        # 2. Groq Init
        if config.GROQ_API_KEY:
            self.groq_client = Groq(api_key=config.GROQ_API_KEY)
        else:
            logging.warning("⚠️ Groq API Key is missing.")
            self.groq_client = None

    def get_embedding(self, text: str) -> List[float]:
        """텍스트 임베딩 생성"""
        if not self.openai_client or not text:
            return [0.0] * 1536  # Mock return on failure

        try:
            text = text.replace("\n", " ")
            response = self.openai_client.embeddings.create(
                input=[text], 
                model=config.EMBEDDING_MODEL
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"Embedding Error: {e}")
            return [0.0] * 1536

    def chat_fast(self, system_prompt: str, user_prompt: str) -> str:
        """[System 1] Groq: 빠른 추론"""
        if not self.groq_client:
            return "(Groq N/A)"

        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=config.FAST_MODEL,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Groq Chat Error: {e}")
            return "(Fast Inference Failed)"

    def chat_slow(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> Union[str, dict]:
        """[System 2] GPT-4o: 고지능 추론"""
        if not self.openai_client:
            return {} if json_mode else "(OpenAI N/A)"

        try:
            params = {
                "model": config.SMART_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.8,
            }
            if json_mode:
                params["response_format"] = {"type": "json_object"}

            response = self.openai_client.chat.completions.create(**params)
            content = response.choices[0].message.content

            if json_mode:
                return json.loads(content)
            return content

        except Exception as e:
            logging.error(f"OpenAI Chat Error: {e}")
            return {} if json_mode else f"(Smart Inference Failed: {e})"