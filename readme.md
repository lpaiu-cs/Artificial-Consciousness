# 🧠 CogBot: 인지 심리학 기반 그룹 챗봇 (Cognitive Group Chatbot)

> **"단순한 검색(RAG)을 넘어, 인간처럼 기억하고 망각하며, 눈치껏 대화하는 AI 에이전트"**

CogBot은 단순한 질의응답 시스템이 아닙니다. 인간의 인지 과정(Cognitive Process)을 모사하여, **단기 기억(STM)**과 **장기 기억(LTM)**을 유기적으로 순환시키는 **이원화된 메모리 아키텍처(Dual-Process Memory Architecture)**를 탑재했습니다.

다자간 채팅 환경(Group Chat)에서 효율적인 토큰 사용과 자연스러운 상호작용을 위해 **System 1(직관/빠른 처리)**과 **System 2(사고/느린 처리)** 모델을 결합하였습니다.

---

## 🌟 핵심 기능 (Key Features)

### 1. 이원화된 메모리 시스템 (Dual-Memory System)

* **System 1 (Fast STM):** 우선순위 큐(Priority Queue)를 사용하여 작업 기억을 관리합니다. 중요하지 않거나 오래된 기억은 자동으로 밀려납니다(Eviction).
* **System 2 (Slow LTM):** 밀려난 기억들을 버리지 않고 모아서 비동기적으로 회고(Consolidation)합니다. LLM이 이를 분석해 '불변의 사실(Fact)'과 '추억(Episode)'으로 정제하여 영구 저장합니다.

### 2. 비용 효율적인 중요도 평가 (Cost-Effective Scoring)

* 모든 메시지에 LLM을 사용하지 않습니다. **NLP 기반의 경량화된 Scorer**가 0.01초 만에 메시지의 중요도(감정, 정보성, 호출 여부 등)를 채점합니다.
* "ㅋㅋ", "ㅇㅇ" 같은 단순 반응은 빠르게 망각되고, 중요한 정보는 오랫동안 단기 기억에 생존합니다.

### 3. 눈치 챙기기 & 끼어들기 (Smart Interjection)

* 봇은 호출되지 않아도 대화 맥락을 읽습니다.
* 할 말이 없거나 끼어들 타이밍이 아니라고 판단되면 `[PASS]` 토큰을 발행하여 침묵합니다. 이를 통해 "눈치 없는 봇" 문제를 해결했습니다.

### 4. 심층 페르소나 (Deep Persona)

* 1,000자 이상의 상세한 페르소나 정의를 통해 말투, 성격, 금기 사항 등을 일관되게 유지합니다. 단순한 프롬프트 지시를 넘어선 캐릭터 몰입감을 제공합니다.

---

## 🏗️ 시스템 아키텍처 (Architecture)

CogBot은 인간의 기억 처리 과정을 공학적으로 모델링했습니다.

```mermaid
raph TD
    User[사용자 입력] --> |NLP Scoring| Scorer{중요도 채점}
    Scorer --> |Score & Content| STM["단기 기억 (Priority Queue)"]
    
    STM -- "용량 초과 (Eviction)" --> Buffer[망각 버퍼]
    Buffer -- "주기적 실행 (Batch)" --> LTM_Manager["LTM 관리자 (System 2)"]
    
    LTM_Manager --> |"LLM 회고 (Consolidation)"| Facts["의미 기억 (Facts)"]
    LTM_Manager --> |"LLM 회고 (Consolidation)"| Episodes["일화 기억 (Episodes)"]
    
    STM & Facts & Episodes --> |Context Injection| LLM["LLM (답변 생성)"]
    LLM --> |Response / PASS| User

```

1. **감각 등록 (Sensory):** 메시지가 들어오면 `ImportanceScorer`가 즉시 점수(1.0~10.0)를 매깁니다.
2. **작업 기억 (Working Memory):** 점수에 따라 `FastSTM` 큐에 정렬됩니다. 용량이 차면 점수가 낮은 기억부터 방출됩니다.
3. **기억 공고화 (Consolidation):** 방출된 기억은 `SlowLTM`이 수거하여, LLM을 통해 핵심 정보만 장기 기억으로 승격시킵니다.
4. **인출 및 생성 (Retrieval & Generation):** 답변 시, 현재의 `STM`과 관련 있는 `LTM`을 프롬프트에 주입하여 답변을 생성합니다.

---

## 📂 프로젝트 구조 (Directory Structure)

```bash
CogBot/
├── config.py               # API Key, 페르소나, 모델 설정
├── llm_handler.py          # 메인 컨트롤러 (입출력 및 오케스트레이션)
├── memory_handler.py       # FastSTM(단기) 및 SlowLTM(장기) 클래스 구현
├── memory_structures.py    # MemoryObject, ImportanceScorer(NLP) 구현
├── ltm_storage.json        # 장기 기억 데이터베이스 (자동 생성)
├── requirements.txt        # 의존성 패키지 목록
└── README.md               # 문서

```

---

## 🚀 설치 및 시작 (Getting Started)

### 1. 사전 요구 사항 (Prerequisites)

* Python 3.8 이상
* OpenAI API Key (또는 호환되는 LLM API Key)

### 2. 설치 (Installation)

```bash
# 레포지토리 클론
git clone https://github.com/your-username/CogBot.git
cd CogBot

# 의존성 설치
pip install -r requirements.txt

```

### 3. 설정 (Configuration)

`config.py` 파일을 생성하고 다음과 같이 설정합니다.

```python
# config.py

LLM_API_KEY = "sk-your-openai-api-key-here"
LLM_MODEL = "gpt-4o"  # 또는 gpt-3.5-turbo
BOT_USER_ID = "999"   # 봇의 식별 ID

# 1000자 이상의 상세 페르소나
BOT_PERSONA = """
[이름: 잼봇]
너는 이 채팅방의 분위기 메이커이자 든든한 친구야.
... (생략: 상세한 성격, 말투, 행동 지침) ...
"""

```

### 4. 실행 (Usage)

메인 스크립트나 챗봇 인터페이스에서 `LLMHandler`를 인스턴스화하여 사용합니다.

```python
from llm_handler import LLMHandler

bot = LLMHandler()

# 대화 기록 예시 (실제로는 카카오톡/디스코드 API에서 받아옴)
history = [
    {"user_id": "user1", "msg": "안녕, 오늘 기분 어때?"},
    {"user_id": "user2", "msg": "나 오늘 좀 우울해.."}
]

# 답변 생성 요청
response = bot.get_response(history)

if response:
    print(f"Bot: {response}")
else:
    print("(봇이 눈치껏 침묵했습니다)")

```

---

## 🧠 기술 상세 (Technical Deep Dive)

### 1. 우선순위 큐 기반 망각 (Priority Queue Eviction)

`memory_structures.py`의 `MemoryObject`는 다음 공식으로 **보존 점수(Retention Score)**를 계산합니다.

* **Recency:** 시간이 지날수록 0에 수렴 (에빙하우스 망각 곡선 응용).
* **Importance:** 감정 단어, 봇 호출, 정보성 키워드 등이 있으면 높음.
* **결과:** "중요한 정보"는 오래되어도 큐에 남고, "가벼운 잡담"은 최신이라도 금방 밀려납니다.

### 2. 비동기 회고 (Asynchronous Reflection)

대화의 응답 속도(Latency)를 저해하지 않기 위해, 장기 기억 저장 프로세스는 `threading`을 사용하여 백그라운드에서 실행됩니다.

* 단기 기억에서 밀려난 데이터가 버퍼에 5개 이상 쌓이면 `SlowLTM.consolidate()`가 트리거됩니다.
* LLM은 이 파편들을 분석하여 `{"facts": [...], "episodes": [...]}` 형태의 JSON으로 추출합니다.

---

## 🔮 향후 계획 (Roadmap)

* [ ] **Vector DB 도입:** 장기 기억 데이터가 방대해질 경우를 대비해 FAISS 또는 ChromaDB 연동.
* [ ] **선톡 스케줄러:** 사용자가 오랫동안 말이 없을 때, 기억을 바탕으로 먼저 안부를 묻는 기능 추가.
* [ ] **감정 분석 고도화:** 현재의 키워드 매칭 방식을 넘어선 딥러닝 기반 감정 분석 모델 적용.
