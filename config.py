import os

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Model Config
EMBEDDING_MODEL = "text-embedding-3-small"
SMART_MODEL = "gpt-4o"
FAST_MODEL = "llama3-70b-8192"

# Memory Config
STM_CAPACITY = 15
LTM_SAVE_PATH = "memory_graph.json"
SOCIAL_SAVE_PATH = "social_map.json"
REFLECTION_INTERVAL = 30  # 초

# Scoring Thresholds
MIN_ACTIVATION = 10.0
BOOST_SCORE = 20.0
PENALTY_SCORE = -5.0
TIME_DECAY = 2.0