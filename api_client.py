import logging
import json
import time
import os
from typing import List, Any, Union
from datetime import datetime

# 외부 라이브러리
from openai import OpenAI
from groq import Groq

# 설정 로드
import config


class APILogger:
    """
    [API Request/Response Logger]
    API 호출을 JSONL 형식으로 기록합니다.
    테스트 시 config.API_LOGGING_ENABLED = False로 비활성화 가능.
    """
    
    def __init__(self, log_file: str = None, enabled: bool = None, exclude_embedding: bool = None):
        # enabled가 명시적으로 False면 로깅 비활성화
        if enabled is False:
            self.enabled = False
            self.log_file = None
            self.max_content_length = 500
            self.log_level = "WARNING"
            self.exclude_embedding = True
            return
            
        self.log_file = log_file or getattr(config, 'API_LOG_FILE', 'api_logs.jsonl')
        self.enabled = enabled if enabled is not None else getattr(config, 'API_LOGGING_ENABLED', True)
        self.max_content_length = getattr(config, 'API_LOG_MAX_CONTENT_LENGTH', 500)
        self.log_level = getattr(config, 'API_LOG_LEVEL', 'DEBUG')
        self.exclude_embedding = exclude_embedding if exclude_embedding is not None else getattr(config, 'API_LOG_EXCLUDE_EMBEDDING', False)
        
    def set_enabled(self, enabled: bool):
        """로깅 on/off 토글"""
        self.enabled = enabled
    
    def set_exclude_embedding(self, exclude: bool):
        """Embedding 로깅 제외 토글"""
        self.exclude_embedding = exclude
        
    def _truncate(self, text: str, max_len: int = None) -> str:
        """긴 텍스트 자르기"""
        max_len = max_len or self.max_content_length
        if text and len(text) > max_len:
            return text[:max_len] + f"... [truncated, total {len(text)} chars]"
        return text
    
    def _should_log(self, level: str) -> bool:
        """로그 레벨 체크"""
        if not self.enabled:
            return False
        levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
        return levels.get(level, 0) >= levels.get(self.log_level, 0)
    
    def log(self, entry: dict, level: str = "DEBUG"):
        """로그 엔트리 기록"""
        if not self._should_log(level):
            return
            
        entry["timestamp"] = datetime.now().isoformat()
        entry["level"] = level
        
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logging.error(f"API Log Write Error: {e}")
    
    def log_embedding_request(self, text: str, model: str):
        """임베딩 요청 로그"""
        if self.exclude_embedding:
            return
        self.log({
            "type": "EMBEDDING_REQUEST",
            "model": model,
            "input_text": self._truncate(text),
            "input_length": len(text) if text else 0
        }, level="DEBUG")
    
    def log_embedding_response(self, text: str, embedding_dim: int, duration_ms: float, success: bool, error: str = None):
        """임베딩 응답 로그"""
        if self.exclude_embedding:
            return
        entry = {
            "type": "EMBEDDING_RESPONSE",
            "input_preview": self._truncate(text, 100),
            "embedding_dim": embedding_dim,
            "duration_ms": round(duration_ms, 2),
            "success": success
        }
        if error:
            entry["error"] = str(error)
        self.log(entry, level="INFO" if success else "ERROR")
    
    def log_chat_request(self, provider: str, model: str, system_prompt: str, user_prompt: str, json_mode: bool = False):
        """채팅 요청 로그"""
        self.log({
            "type": "CHAT_REQUEST",
            "provider": provider,
            "model": model,
            "system_prompt": self._truncate(system_prompt),
            "user_prompt": self._truncate(user_prompt),
            "json_mode": json_mode,
            "system_prompt_length": len(system_prompt) if system_prompt else 0,
            "user_prompt_length": len(user_prompt) if user_prompt else 0
        }, level="DEBUG")
    
    def log_chat_response(self, provider: str, model: str, response: str, duration_ms: float, 
                          success: bool, token_usage: dict = None, error: str = None):
        """채팅 응답 로그"""
        entry = {
            "type": "CHAT_RESPONSE",
            "provider": provider,
            "model": model,
            "response": self._truncate(str(response)),
            "response_length": len(str(response)) if response else 0,
            "duration_ms": round(duration_ms, 2),
            "success": success
        }
        if token_usage:
            entry["token_usage"] = token_usage
        if error:
            entry["error"] = str(error)
        self.log(entry, level="INFO" if success else "ERROR")


class UnifiedAPIClient:
    """
    [API Gateway]
    OpenAI와 Groq 클라이언트를 통합 관리합니다.
    """
    def __init__(self, enable_logging: bool = None, exclude_embedding_log: bool = None):
        # Logger 초기화
        self.logger = APILogger(enabled=enable_logging, exclude_embedding=exclude_embedding_log)
        
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
    
    def set_logging(self, enabled: bool):
        """로깅 on/off (테스트용)"""
        self.logger.set_enabled(enabled)
    
    def set_exclude_embedding_log(self, exclude: bool):
        """Embedding 로깅 제외 토글"""
        self.logger.set_exclude_embedding(exclude)

    def get_embedding(self, text: str) -> List[float]:
        """텍스트 임베딩 생성"""
        if not self.openai_client or not text:
            return [0.0] * 1536  # Mock return on failure

        # 로그: 요청
        self.logger.log_embedding_request(text, config.EMBEDDING_MODEL)
        start_time = time.time()
        
        try:
            text = text.replace("\n", " ")
            response = self.openai_client.embeddings.create(
                input=[text], 
                model=config.EMBEDDING_MODEL
            )
            embedding = response.data[0].embedding
            
            # 로그: 성공 응답
            duration_ms = (time.time() - start_time) * 1000
            self.logger.log_embedding_response(text, len(embedding), duration_ms, success=True)
            
            return embedding
        except Exception as e:
            # 로그: 에러
            duration_ms = (time.time() - start_time) * 1000
            self.logger.log_embedding_response(text, 0, duration_ms, success=False, error=str(e))
            logging.error(f"Embedding Error: {e}")
            return [0.0] * 1536

    def chat_fast(self, system_prompt: str, user_prompt: str) -> str:
        """[System 1] Groq: 빠른 추론"""
        if not self.groq_client:
            return "(Groq N/A)"

        # 로그: 요청
        self.logger.log_chat_request("Groq", config.FAST_MODEL, system_prompt, user_prompt)
        start_time = time.time()
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=config.FAST_MODEL,
                temperature=0.3,
            )
            content = response.choices[0].message.content
            
            # 로그: 성공 응답
            duration_ms = (time.time() - start_time) * 1000
            token_usage = None
            if hasattr(response, 'usage') and response.usage:
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            self.logger.log_chat_response("Groq", config.FAST_MODEL, content, duration_ms, 
                                          success=True, token_usage=token_usage)
            
            return content
        except Exception as e:
            # 로그: 에러
            duration_ms = (time.time() - start_time) * 1000
            self.logger.log_chat_response("Groq", config.FAST_MODEL, None, duration_ms, 
                                          success=False, error=str(e))
            logging.error(f"Groq Chat Error: {e}")
            return "(Fast Inference Failed)"

    def chat_slow(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> Union[str, dict]:
        """[System 2] GPT-4o: 고지능 추론"""
        if not self.openai_client:
            return {} if json_mode else "(OpenAI N/A)"

        # 로그: 요청
        self.logger.log_chat_request("OpenAI", config.SMART_MODEL, system_prompt, user_prompt, json_mode)
        start_time = time.time()
        
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
            
            # 로그: 성공 응답
            duration_ms = (time.time() - start_time) * 1000
            token_usage = None
            if hasattr(response, 'usage') and response.usage:
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            result = json.loads(content) if json_mode else content
            self.logger.log_chat_response("OpenAI", config.SMART_MODEL, content, duration_ms,
                                          success=True, token_usage=token_usage)
            
            return result

        except Exception as e:
            # 로그: 에러
            duration_ms = (time.time() - start_time) * 1000
            self.logger.log_chat_response("OpenAI", config.SMART_MODEL, None, duration_ms,
                                          success=False, error=str(e))
            logging.error(f"OpenAI Chat Error: {e}")
            return {} if json_mode else f"(Smart Inference Failed: {e})"