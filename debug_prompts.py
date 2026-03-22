import time

from cogbot.bot_orchestrator import BotOrchestrator
from cogbot.memory_structures import EpisodeNode

# =========================================================
# 1. 디버깅용 가짜 API 클라이언트 (Interceptor)
# =========================================================
class DebugAPIClient:
    """
    실제 API를 호출하는 대신, 요청 내용을 콘솔에 예쁘게 출력해주는 클래스입니다.
    """
    def __init__(self):
        print("🔧 [DebugMode] API Client Initialized (No Real Cost)")

    def get_embedding(self, text):
        # 임베딩 요청은 너무 빈번하므로 짧게 로그만 남기고 더미 벡터 반환
        # print(f"   [Embedding] '{text[:20]}...' (Len: {len(text)})")
        return [0.0] * 1536

    def chat_fast(self, system_prompt, user_prompt):
        print("\n" + "="*60)
        print(f"⚡ [System 1: Groq Request] (상황 요약 요청)")
        print("="*60)
        print(f"📌 [System Prompt]:\n{system_prompt}")
        print("-" * 30)
        print(f"📌 [User Prompt]:\n{user_prompt}")
        print("="*60)
        
        # 가짜 응답 반환 (봇이 다음 단계로 넘어가게 하기 위해)
        return "상황 요약: 사용자가 인사를 건넸고, 봇은 이에 대해 적절히 응답해야 함."

    def chat_slow(self, system_prompt, user_prompt, json_mode=False):
        print("\n" + "="*60)
        print(f"🧠 [System 2: OpenAI Request] (최종 답변 생성)")
        print("="*60)
        print(f"📌 [System Prompt]:\n{system_prompt}")
        print("-" * 30)
        print(f"📌 [User Prompt]:\n{user_prompt}")
        print("="*60)
        
        # 가짜 응답 반환 (감정 태그 포함)
        return "안녕! 테스트 중이구나. [FEELING:차분한 호기심]"

# =========================================================
# 2. 봇 초기화 및 의존성 주입
# =========================================================
def run_debug_scenario(case_name, history, current_msg):
    print(f"\n\n🚀 [Scenario: {case_name}] 시작...")
    
    # 봇 생성
    bot = BotOrchestrator()
    
    # [핵심] 봇이 내부적으로 쓰는 API 클라이언트를 위에서 만든 'DebugAPIClient'로 바꿔치기
    debug_api = DebugAPIClient()
    bot.api = debug_api
    bot.social.api = debug_api
    bot.sensory.api = debug_api
    bot.ltm.api = debug_api # LTM 검색 시 임베딩 때문에 필요
    
    # LTM 검색 결과도 임의로 고정 (테스트 변수 통제)
    # 실제 DB를 뒤지면 복잡하니까, "과거 기억이 하나 검색되었다"고 가정
    mock_memory = EpisodeNode(
        content="사용자는 민트초코를 좋아한다고 말했었음.",
        user_id=current_msg["user_id"],
        emotion_tag="즐거움"
    )
    # retrieve 함수를 강제로 덮어씌워 가짜 기억 반환
    bot.ltm.retrieve = lambda query, top_k: [mock_memory]

    # 실행! (여기서 위의 chat_fast, chat_slow가 호출되며 로그가 찍힘)
    response = bot.process_trigger(history, current_msg)
    
    print(f"\n✅ [Final Response]: {response}")

# =========================================================
# 3. 테스트 케이스 정의 (여기를 수정해서 확인하세요!)
# =========================================================

if __name__ == "__main__":
    
    # --- Case 1: 대화 이력이 조금 있는 상황 ---
    history_data = [
        {"role": "user", "msg": "안녕 코봇, 너 뭐하고 있어?", "user_id": "user1", "user_name": "김철수", "timestamp": time.time()-100},
        {"role": "assistant", "msg": "멍하니 있었어. 너는?", "user_id": "bot", "user_name": "코봇", "timestamp": time.time()-90}
    ]
    
    current_message = {
        "user_id": "user1",
        "user_name": "김철수",
        "msg": "나? 그냥 코드 짜고 있어. 근데 좀 어렵네."
    }
    
    run_debug_scenario("기본 대화 흐름 확인", history_data, current_message)

    
    # --- Case 2: 닉네임이 바뀐 상황 (ID Rendering 확인) ---
    # 사용자가 '김철수' -> '파괴왕'으로 닉네임을 바꿨을 때 프롬프트에 어떻게 나오는지 확인
    current_message_renamed = {
        "user_id": "user1",
        "user_name": "파괴왕", # 닉네임 변경!
        "msg": "야 나 닉네임 바꿨는데 어때?"
    }
    
    # 이전 history는 그대로 김철수로 남아있음
    run_debug_scenario("닉네임 변경 시점 확인", history_data, current_message_renamed)
