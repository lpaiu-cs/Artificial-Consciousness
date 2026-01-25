# api_client.py
import os
import json
import logging
from typing import List, Optional, Dict, Any

# 외부 라이브러리 (설치 필요)
from openai import OpenAI
from groq import Groq
import config  # config.py에 API KEY들이 있다고 가정

class UnifiedAPIClient:
    """
    [API Gateway]
    - OpenAI (GPT-4o, Embedding)
    - Groq (Llama3-70b/8b for Fast Inference)
    모든 외부 모델 호출은 이 클래스를 통해 이루어집니다.
    """
    def __init__(self):
        # 1. OpenAI Client (Intelligence & Embedding)
        try:
            self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
            self.embedding_model = "text-embedding-3-small"
            self.smart_model = "gpt-4o"
        except Exception as e:
            logging.error(f"OpenAI Client Init Failed: {e}")
            self.openai_client = None

        # 2. Groq Client (Speed)
        try:
            self.groq_client = Groq(api_key=config.GROQ_API_KEY)
            self.fast_model = "llama3-70b-8192" # 혹은 llama3-8b-8192
        except Exception as e:
            logging.error(f"Groq Client Init Failed: {e}")
            self.groq_client = None

    def get_embedding(self, text: str) -> List[float]:
        """텍스트를 벡터로 변환 (기존 vector_engine 대체)"""
        if not self.openai_client or not text:
            return [0.0] * 1536 # Mock for failure

        try:
            text = text.replace("\n", " ")
            response = self.openai_client.embeddings.create(
                input=[text], 
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"Embedding Error: {e}")
            return [0.0] * 1536

    def chat_fast(self, system_prompt: str, user_prompt: str) -> str:
        """
        [System 1] Groq를 사용한 빠른 응답/요약
        """
        if not self.groq_client:
            return "(Groq Client Missing) API 키를 확인하세요."

        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.fast_model,
                temperature=0.3, # 사실적 요약 위주
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Groq Chat Error: {e}")
            return "(Fast Inference Failed)"

    def chat_slow(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> Any:
        """
        [System 2] GPT-4o를 사용한 고지능 응답/성찰
        json_mode=True일 경우 파싱된 dict를 반환합니다.
        """
        if not self.openai_client:
            return {} if json_mode else "(OpenAI Client Missing)"

        try:
            params = {
                "model": self.smart_model,
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
            else:
                return content

        except Exception as e:
            logging.error(f"OpenAI Chat Error: {e}")
            return {} if json_mode else f"(Smart Inference Failed: {e})"