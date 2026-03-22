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
SMART_MODEL = os.getenv("SMART_MODEL", "gpt-4o-2024-11-20")
FAST_MODEL = os.getenv("FAST_MODEL", "llama-3.1-8b-instant")
# FAST_MODEL = "llama-3.1-70b-versatile"
MODEL_EVAL_GATE_ENABLED = os.getenv("MODEL_EVAL_GATE_ENABLED", "true").lower() not in {"0", "false", "no", "off"}
MODEL_EVAL_GATE_ENFORCE = os.getenv("MODEL_EVAL_GATE_ENFORCE", "true").lower() not in {"0", "false", "no", "off"}
MODEL_EVAL_GATE_PATH = os.getenv("MODEL_EVAL_GATE_PATH", "model_eval_gate.json")

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
CANONICAL_ENCRYPTION_KEY = os.getenv("CANONICAL_ENCRYPTION_KEY", "")  # legacy 단일 키 호환용
CANONICAL_ENCRYPTION_KEY_ID = os.getenv("CANONICAL_ENCRYPTION_KEY_ID", "local-v1")  # legacy 단일 키 ID
CANONICAL_ACTIVE_KEY_ID = os.getenv("CANONICAL_ACTIVE_KEY_ID", CANONICAL_ENCRYPTION_KEY_ID)
CANONICAL_ENCRYPTION_KEYS_JSON = os.getenv("CANONICAL_ENCRYPTION_KEYS_JSON", "")  # {"key_id": "fernet_key"}
CANONICAL_ENCRYPTION_KEYRING_PATH = os.getenv("CANONICAL_ENCRYPTION_KEYRING_PATH", "")  # 비어 있으면 DB 옆 keyring sidecar 생성
CANONICAL_ENCRYPTION_KEY_PATH = os.getenv("CANONICAL_ENCRYPTION_KEY_PATH", "")  # legacy raw key file migration용
CANONICAL_AUTO_REENCRYPT_PROTECTED_PAYLOADS = os.getenv("CANONICAL_AUTO_REENCRYPT_PROTECTED_PAYLOADS", "false").lower() not in {"0", "false", "no", "off"}
SENSORY_CURSOR_PATH = "sensory_seen_logs.json"  # delta ingest cursor 저장소
SENSORY_CURSOR_MAX_ENTRIES = 50000              # durable cursor 최대 유지 개수
SENSORY_CURSOR_HMAC_KEY = os.getenv("SENSORY_CURSOR_HMAC_KEY", "")  # cursor digest용 선택적 비밀키
BOUNDARY_DEDUPE_MAX_ENTRIES = 2048             # in-memory boundary dedupe 최대 유지 개수
REFERENT_CACHE_MAX_ENTRIES = 6                 # 세션 referent cache 최대 유지 개수
BOUNDARY_SEMANTIC_MATCH_THRESHOLD = 0.78       # boundary 의미 유사도 기반 redaction 임계치
BOUNDARY_SEMANTIC_MAX_CANDIDATES = 4           # 턴당 semantic fallback 비교 후보 수 상한
BOUNDARY_SEGMENT_EMBED_CACHE_MAX = 256         # segment embedding cache 최대 유지 개수
BOUNDARY_SEMANTIC_RESULT_CACHE_MAX = 2048      # semantic match result cache 최대 유지 개수

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
