import os
from dotenv import load_dotenv

load_dotenv()

# Bot self ID
BOT_USER_ID = "436468165"

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Model Config
EMBEDDING_MODEL = "text-embedding-3-small"
SMART_MODEL = "gpt-4o"
FAST_MODEL = "llama-3.1-8b-instant"
# FAST_MODEL = "llama-3.1-70b-versatile"

# Memory Config
STM_CAPACITY = 15
LTM_SAVE_PATH = "memory_graph.json"
SOCIAL_SAVE_PATH = "social_map.json"
REFLECTION_INTERVAL = 30  # 초

# Scoring Thresholds (STM)
MIN_ACTIVATION = 10.0
BOOST_SCORE = 20.0
PENALTY_SCORE = -5.0
TIME_DECAY = 2.0

# === LTM Retrieval Weights ===
# Phase 1: Anchoring (Vector Search)
CLAIM_BONUS = 1.35           # Claim 노드 검색 시 보너스 가중치
INSIGHT_BONUS = 1.2          # Insight 노드 검색 시 보너스 가중치
NOTE_BASE_WEIGHT = 0.9       # Note 노드 기본 가중치
EPISODE_BASE_WEIGHT = 1.0    # Episode 노드 기본 가중치

# Phase 2: Spreading (Graph Traversal)
SPREAD_DECAY_FACTOR = 0.8    # 그래프 확산 시 감쇠 계수
INSIGHT_TO_EPISODE_BOOST = 1.1  # Insight -> Episode 확산 보너스
EPISODE_TO_EPISODE_DECAY = 0.9  # Episode -> Episode 확산 감쇠

# Phase 3: Reranking & Contextual Adjustment
MOOD_CONGRUENCE_BOOST = 1.2  # 기분 일치 시 점수 증폭
RECENCY_DECAY_RATE = 0.05    # 시간 경과에 따른 감쇠율 (per 24h)
KEYWORD_MATCH_BOOST = 1.5    # 키워드 매칭 시 점수 증폭

# === Graph Edge Weights ===
TEMPORAL_EDGE_FORWARD = 0.5   # Episode -> 다음 Episode 연결 강도
TEMPORAL_EDGE_BACKWARD = 1.0  # Episode <- 이전 Episode 연결 강도 (역방향)
EVIDENCE_EDGE_TO_EPISODE = 0.8   # Insight -> Episode 연결 강도
EVIDENCE_EDGE_TO_INSIGHT = 1.0   # Episode -> Insight 연결 강도

# === Storage Paths ===
LTM_GRAPH_PATH = "ltm_graph.json"       # 그래프 구조 (노드 메타데이터 + 엣지)
LTM_EMBEDDINGS_PATH = "ltm_embeddings.json"  # 임베딩 벡터 저장소
CANONICAL_MEMORY_DB_PATH = "memory_state.sqlite3"  # 정본 상태 저장소 (SQLite)
SENSORY_CURSOR_PATH = "sensory_seen_logs.json"  # delta ingest cursor 저장소
SENSORY_CURSOR_MAX_ENTRIES = 50000              # durable cursor 최대 유지 개수

# Social Update Config
# 이 감정과 가까울수록 호감도가 오르고, 멀수록(반대일수록) 호감도가 떨어집니다.
POSITIVE_EMOTION_ANCHOR = "joyful trust and happiness" 

# 호감도 변화 스케일링 (유사도 1.0일 때 최대 몇 점 변할지)
SOCIAL_SENSITIVITY = 5.0

# 봇 자아
BOT_NAME = "코봇"

# === API Logging Config ===
API_LOGGING_ENABLED = False         # 민감 데이터가 기본으로 기록되지 않도록 안전 기본값 적용
API_LOG_FILE = "api_logs.jsonl"     # 로그 파일 경로
API_LOG_LEVEL = "WARNING"           # WARNING 이상만 기본 기록
API_LOG_INCLUDE_CONTENT = False     # 기본적으로 프롬프트/응답 원문은 기록하지 않음
API_LOG_MAX_CONTENT_LENGTH = 500    # 로그에 기록할 최대 콘텐츠 길이 (truncate)
API_LOG_EXCLUDE_EMBEDDING = False   # Embedding 로그 제외 (True면 채팅만 기록)
