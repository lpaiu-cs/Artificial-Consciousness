import json
import os
import requests
import threading
from datetime import datetime
import config

class MemoryManager:
    """
    유저의 장기 기억(프로필 + 에피소드)을 전문적으로 관리하는 매니저.
    스타일 분석 로직은 제거되었으며, '팩트'와 '추억'을 적재하는 데 집중함.
    """
    def __init__(self, db_path="user_memory.json"):
        self.db_path = db_path
        # 메모리 구조: { user_id: { "profile": {...}, "episodes": [...] } }
        self.memories = self._load_db()

    def _load_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_db(self):
        try:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self.memories, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 메모리 저장 실패: {e}")

    def get_relevant_context(self, active_users):
        """
        현재 대화에 참여 중인 유저들의 기억을 LLM이 읽기 좋은 텍스트로 변환.
        [전략]
        1. 유저의 '기본 프로필(성향)'은 항상 가져옴.
        2. 유저와의 '추억(에피소드)'은 최신순으로 3개만 가져와서 문맥에 띄움.
        """
        context_lines = []
        for uid in active_users:
            uid = str(uid)
            mem = self.memories.get(uid)
            if not mem:
                continue

            # 1. 의미 기억 (Profile)
            profile = mem.get("profile", {})
            profile_str = ", ".join([f"{k}: {v}" for k, v in profile.items()])
            
            # 2. 일화 기억 (Episodes) - 최근 3개
            episodes = mem.get("episodes", [])
            recent_episodes = episodes[-3:] if episodes else []
            episode_str = ""
            if recent_episodes:
                episode_text = "\n    ".join([f"- [{e['date']}] {e['summary']}" for e in recent_episodes])
                episode_str = f"\n    (함께 했던 추억):\n    {episode_text}"

            context_lines.append(f"👤 **User {uid} 정보**\n    - 특징: {profile_str}{episode_str}")

        return "\n".join(context_lines) if context_lines else "특이 사항 없음."

    def reflect_async(self, recent_history):
        """
        [비동기 회고]
        대화 로그를 분석하여 '새로운 사실(Update)'과 '새로운 에피소드(Append)'를 분리하여 저장.
        """
        def _task():
            try:
                # 봇의 메시지는 기억 분석 대상에서 제외 (유저의 행동만 기억)
                filtered_logs = [
                    f"{h.get('user_name', 'User')}({h['user_id']}): {h['msg']}" 
                    for h in recent_history 
                    if str(h.get('user_id')) != str(config.BOT_USER_ID)
                ]
                
                if not filtered_logs: return
                log_text = "\n".join(filtered_logs)
                today_date = datetime.now().strftime("%Y-%m-%d")

                # 프롬프트: 사실(Fact)과 에피소드(Episode)를 구분하여 추출
                prompt = f"""
                너는 '기억 관리자'다. 아래 채팅 로그를 분석해서 각 유저에 대한 정보를 JSON으로 추출해라.
                
                [분석 규칙]
                1. **profile**: 변하지 않는 사실이나 현재 상태 (직업, MBTI, 취미, 거주지, 좋아하는 것 등). 
                   - 기존 정보를 덮어쓰거나 병합할 수 있는 짧은 키워드 위주.
                2. **new_episode**: 오늘 대화에서 특별히 기억해야 할 '사건'이나 '감정적인 순간'. 
                   - 예: "철수가 여자친구와 헤어져서 다들 위로해줌", "영희가 새 프로젝트를 시작했다고 자랑함".
                   - 별거 없는 일상적인 대화(인사 등)라면 빈칸으로 남겨라. **정말 중요한 추억만 기록해.**
                
                [현재 날짜] {today_date}

                [출력 포맷 (JSON only)]
                {{
                    "user_id_1": {{
                        "profile": {{"직업": "개발자", "취미": "게임"}},
                        "new_episode": "오늘 야근 때문에 매우 힘들어했음."
                    }},
                    "user_id_2": ...
                }}
                
                [대화 로그]
                {log_text}
                """

                payload = {
                    "model": config.LLM_MODEL, 
                    "messages": [{"role": "system", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.5
                }
                
                headers = {"Authorization": f"Bearer {config.LLM_API_KEY}", "Content-Type": "application/json"}
                r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=20)
                
                if r.status_code == 200:
                    result = json.loads(r.json()['choices'][0]['message']['content'])
                    self._update_memory_structure(result, today_date)
                    
            except Exception as e:
                print(f"🧠 기억 회고 중 오류 발생: {e}")

        # 메인 스레드 영향 없도록 백그라운드 실행
        threading.Thread(target=_task).start()

    def _update_memory_structure(self, extracted_data, date_str):
        """추출된 데이터를 실제 DB 구조에 반영"""
        changed = False
        for uid, data in extracted_data.items():
            uid = str(uid)
            if uid not in self.memories:
                self.memories[uid] = {"profile": {}, "episodes": []}
            
            # 1. 프로필 업데이트 (기존 키가 있으면 덮어쓰기/추가)
            if "profile" in data and isinstance(data["profile"], dict):
                for k, v in data["profile"].items():
                    # 정보가 구체적이거나 새로운 경우 업데이트
                    self.memories[uid]["profile"][k] = v
                    changed = True
            
            # 2. 에피소드 추가 (값이 있을 때만)
            if "new_episode" in data and data["new_episode"]:
                new_ep = {
                    "date": date_str,
                    "summary": data["new_episode"]
                }
                # 중복 방지 (같은 날짜에 완전히 같은 내용이면 스킵)
                if not any(e['summary'] == new_ep['summary'] for e in self.memories[uid]["episodes"]):
                    self.memories[uid]["episodes"].append(new_ep)
                    # 에피소드는 너무 많이 쌓이면 오래된 순으로 삭제 (최대 20개 유지)
                    if len(self.memories[uid]["episodes"]) > 20:
                        self.memories[uid]["episodes"].pop(0)
                    changed = True
                    print(f"✨ [추억 기록] 유저({uid}): {new_ep['summary']}")

        if changed:
            self._save_db()