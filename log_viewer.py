#!/usr/bin/env python
"""
CogBot API Log Viewer
## 📋 로그 뷰어 사용법 요약

이제 로그를 깔끔하게 볼 수 있는 CLI 도구 log_viewer.py가 준비되었습니다:

### 기본 사용법
```bash
python log_viewer.py              # 전체 로그 (compact 모드)
python log_viewer.py -x           # Embedding 로그 제외
python log_viewer.py -d           # 상세 모드
python log_viewer.py --summary    # 통계 요약만 보기
```

### 필터링 옵션
| 옵션 | 설명 |
|------|------|
| `-x, --exclude-embedding` | Embedding 요청/응답 제외 |
| `-p PROVIDER` | 특정 provider만 (OpenAI, Groq) |
| `-e, --errors` | 에러 로그만 |
| `-r, --responses` | 응답 로그만 (요청 제외) |
| `-n N` | 최근 N개만 표시 |
| `-d, --detailed` | 상세 모드 (프롬프트/응답 전체) |
| `-s, --summary` | 통계 요약 |
| `--no-color` | 색상 없이 출력 |

### 조합 예시
```bash
python log_viewer.py -x -d -n 10         # Embedding 제외, 상세모드, 최근 10개
python log_viewer.py -p OpenAI --summary # OpenAI만 + 통계
python log_viewer.py -e                   # 에러만 보기
```

### 런타임 설정 (코드에서)
```python
api_client.set_exclude_embedding_log(True)   # Embedding 로깅 끄기
api_client.set_logging(False)                # 전체 로깅 끄기
```

### 설정 파일 (config.py)
- `API_LOG_EXCLUDE_EMBEDDING = True` → 기본적으로 Embedding 로그 제외

Made changes.
"""
import json
import argparse
import os
import sys
from datetime import datetime
from typing import List, Dict, Optional

# 색상 코드 (터미널용)
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    GRAY = '\033[90m'
    BOLD = '\033[1m'
    END = '\033[0m'
    
    @classmethod
    def disable(cls):
        """모든 색상 코드 비활성화"""
        for attr in ['HEADER', 'BLUE', 'CYAN', 'GREEN', 'YELLOW', 'RED', 'GRAY', 'BOLD', 'END']:
            setattr(cls, attr, '')


class APILogViewer:
    """API 로그 뷰어"""
    
    def __init__(self, log_file: str = "api_logs.jsonl"):
        self.log_file = log_file
        self.entries: List[Dict] = []
        
    def load(self) -> bool:
        """로그 파일 로드"""
        if not os.path.exists(self.log_file):
            print(f"{Colors.RED}❌ 로그 파일이 없습니다: {self.log_file}{Colors.END}")
            return False
            
        self.entries = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        self.entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return True
    
    def filter(self, 
               exclude_embedding: bool = False,
               provider: Optional[str] = None,
               only_errors: bool = False,
               only_responses: bool = False,
               limit: Optional[int] = None) -> List[Dict]:
        """로그 필터링"""
        filtered = self.entries
        
        # Embedding 제외
        if exclude_embedding:
            filtered = [e for e in filtered if 'EMBEDDING' not in e.get('type', '')]
        
        # Provider 필터
        if provider:
            filtered = [e for e in filtered if e.get('provider', '').lower() == provider.lower()]
        
        # 에러만
        if only_errors:
            filtered = [e for e in filtered if e.get('success') == False]
        
        # Response만
        if only_responses:
            filtered = [e for e in filtered if 'RESPONSE' in e.get('type', '')]
        
        # 개수 제한
        if limit:
            filtered = filtered[-limit:]
            
        return filtered
    
    def format_timestamp(self, ts: str) -> str:
        """타임스탬프 포맷팅"""
        try:
            dt = datetime.fromisoformat(ts)
            return dt.strftime("%H:%M:%S")
        except:
            return ts[:8]
    
    def format_duration(self, ms: float) -> str:
        """소요시간 포맷팅"""
        if ms < 1000:
            return f"{ms:.0f}ms"
        return f"{ms/1000:.2f}s"
    
    def wrap_text(self, text: str, width: int = 80, indent: str = "    ") -> str:
        """긴 텍스트를 적절히 줄바꿈"""
        if not text:
            return ""
        
        lines = []
        # 기존 줄바꿈 유지
        for paragraph in text.split('\n'):
            if len(paragraph) <= width:
                lines.append(paragraph)
            else:
                # width 길이로 자르기
                while len(paragraph) > width:
                    # 공백 위치 찾기
                    split_pos = paragraph.rfind(' ', 0, width)
                    if split_pos == -1:
                        split_pos = width
                    lines.append(paragraph[:split_pos])
                    paragraph = paragraph[split_pos:].lstrip()
                if paragraph:
                    lines.append(paragraph)
        
        # 첫 줄 제외 indent 추가
        if len(lines) > 1:
            return lines[0] + '\n' + '\n'.join(indent + line for line in lines[1:])
        return lines[0] if lines else ""
    
    def display_compact(self, entries: List[Dict]):
        """컴팩트 뷰 (한 줄씩)"""
        # 헤더
        header = f"{'시간':<10} {'타입':<10} {'Provider':<8} {'모델':<22} {'상태':<6} {'소요시간':<10}"
        print(f"\n{Colors.BOLD}{header}{Colors.END}")
        print("-" * 70)
        
        for e in entries:
            time_str = self.format_timestamp(e.get('timestamp', ''))
            entry_type = e.get('type', 'UNKNOWN')
            provider = e.get('provider', '-')
            model = (e.get('model', '-') or '-')[:20]
            
            # 타입 표시 (색상 없는 기본값)
            if 'EMBEDDING' in entry_type:
                type_display = "EMBED"
                type_colored = f"{Colors.GRAY}{type_display}{Colors.END}"
            elif 'CHAT' in entry_type:
                type_display = "CHAT"
                type_colored = f"{Colors.CYAN}{type_display}{Colors.END}"
            else:
                type_display = entry_type[:8]
                type_colored = type_display
            
            # 상태 표시
            if 'REQUEST' in entry_type:
                status_display = "REQ"
                status_colored = f"{Colors.BLUE}-> REQ{Colors.END}"
                duration = "-"
            else:
                success = e.get('success', True)
                if success:
                    status_display = "OK"
                    status_colored = f"{Colors.GREEN}OK{Colors.END}"
                else:
                    status_display = "ERR"
                    status_colored = f"{Colors.RED}ERR{Colors.END}"
                duration = self.format_duration(e.get('duration_ms', 0))
            
            # 색상이 비활성화되면 기본 텍스트 사용
            if Colors.END == '':
                type_str = f"{type_display:<10}"
                status_str = f"{status_display:<6}"
            else:
                type_str = f"{type_colored:<19}"  # ANSI 코드 포함 길이 보정
                status_str = f"{status_colored:<15}"
            
            print(f"{time_str:<10} {type_str} {provider:<8} {model:<22} {status_str} {duration:<10}")
    
    def display_detailed(self, entries: List[Dict]):
        """상세 뷰 (카드 형식)"""
        for i, e in enumerate(entries, 1):
            entry_type = e.get('type', 'UNKNOWN')
            
            # 헤더 색상
            if e.get('success') == False:
                header_color = Colors.RED
            elif 'REQUEST' in entry_type:
                header_color = Colors.BLUE
            else:
                header_color = Colors.GREEN
            
            print(f"\n{header_color}{'=' * 80}{Colors.END}")
            print(f"{header_color}[{i}] {entry_type}{Colors.END}")
            print(f"{Colors.GRAY}시간: {e.get('timestamp', 'N/A')}{Colors.END}")
            
            # 공통 정보
            if 'provider' in e:
                print(f"  Provider: {Colors.BOLD}{e['provider']}{Colors.END}")
            if 'model' in e:
                print(f"  모델: {e['model']}")
            
            # Request 정보
            if 'REQUEST' in entry_type:
                if 'system_prompt' in e and e['system_prompt']:
                    sys_text = self.wrap_text(e['system_prompt'], width=76, indent="           ")
                    print(f"  {Colors.GRAY}[System]{Colors.END} {sys_text}")
                if 'user_prompt' in e and e['user_prompt']:
                    user_text = self.wrap_text(e['user_prompt'], width=76, indent="         ")
                    print(f"  {Colors.CYAN}[User]{Colors.END} {user_text}")
                if e.get('json_mode'):
                    print(f"  {Colors.YELLOW}JSON Mode: ON{Colors.END}")
            
            # Response 정보
            if 'RESPONSE' in entry_type:
                duration = self.format_duration(e.get('duration_ms', 0))
                print(f"  소요시간: {Colors.BOLD}{duration}{Colors.END}")
                
                if e.get('success'):
                    if 'response' in e and e['response']:
                        resp_text = self.wrap_text(str(e['response']), width=76, indent="              ")
                        print(f"  {Colors.GREEN}[Response]{Colors.END} {resp_text}")
                    if 'token_usage' in e:
                        t = e['token_usage']
                        print(f"  토큰: {t.get('prompt_tokens', 0)} + {t.get('completion_tokens', 0)} = {Colors.BOLD}{t.get('total_tokens', 0)}{Colors.END}")
                else:
                    print(f"  {Colors.RED}[Error]{Colors.END} {e.get('error', 'Unknown error')}")
    
    def display_summary(self, entries: List[Dict]):
        """통계 요약"""
        if not entries:
            print(f"{Colors.YELLOW}표시할 로그가 없습니다.{Colors.END}")
            return
        
        # 통계 계산
        total = len(entries)
        requests = [e for e in entries if 'REQUEST' in e.get('type', '')]
        responses = [e for e in entries if 'RESPONSE' in e.get('type', '')]
        errors = [e for e in entries if e.get('success') == False]
        
        embedding_reqs = [e for e in entries if 'EMBEDDING' in e.get('type', '')]
        chat_reqs = [e for e in entries if 'CHAT' in e.get('type', '')]
        
        # Provider별
        openai_count = len([e for e in entries if e.get('provider') == 'OpenAI'])
        groq_count = len([e for e in entries if e.get('provider') == 'Groq'])
        
        # 평균 소요시간
        durations = [e.get('duration_ms', 0) for e in responses if e.get('duration_ms')]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # 토큰 합계
        total_tokens = sum(
            e.get('token_usage', {}).get('total_tokens', 0) 
            for e in responses if e.get('token_usage')
        )
        
        print(f"\n{Colors.BOLD}📊 로그 통계 요약{Colors.END}")
        print("─" * 40)
        print(f"  총 로그 수: {Colors.BOLD}{total}{Colors.END}")
        print(f"  요청/응답: {len(requests)} / {len(responses)}")
        print(f"  에러: {Colors.RED if errors else Colors.GREEN}{len(errors)}{Colors.END}")
        print()
        print(f"  {Colors.GRAY}Embedding: {len(embedding_reqs)}{Colors.END}")
        print(f"  {Colors.CYAN}Chat: {len(chat_reqs)}{Colors.END}")
        print()
        print(f"  OpenAI: {openai_count}")
        print(f"  Groq: {groq_count}")
        print()
        print(f"  평균 응답시간: {Colors.BOLD}{self.format_duration(avg_duration)}{Colors.END}")
        print(f"  총 토큰 사용: {Colors.BOLD}{total_tokens:,}{Colors.END}")


def main():
    parser = argparse.ArgumentParser(
        description="CogBot API Log Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python log_viewer.py                    # 기본 뷰
  python log_viewer.py -x                 # Embedding 제외
  python log_viewer.py -d                 # 상세 뷰
  python log_viewer.py -x -n 10           # Embedding 제외, 최근 10개
  python log_viewer.py --provider openai  # OpenAI만 필터
  python log_viewer.py --errors           # 에러만 표시
  python log_viewer.py --summary          # 통계 요약만
        """
    )
    
    parser.add_argument(
        "-f", "--file",
        default="api_logs.jsonl",
        help="로그 파일 경로 (기본: api_logs.jsonl)"
    )
    parser.add_argument(
        "-x", "--exclude-embedding",
        action="store_true",
        help="Embedding 로그 제외"
    )
    parser.add_argument(
        "-p", "--provider",
        choices=["openai", "groq"],
        help="특정 Provider만 필터"
    )
    parser.add_argument(
        "-e", "--errors",
        action="store_true",
        help="에러만 표시"
    )
    parser.add_argument(
        "-r", "--responses",
        action="store_true",
        help="응답만 표시 (요청 제외)"
    )
    parser.add_argument(
        "-n", "--limit",
        type=int,
        help="최근 N개만 표시"
    )
    parser.add_argument(
        "-d", "--detailed",
        action="store_true",
        help="상세 뷰 (카드 형식)"
    )
    parser.add_argument(
        "-s", "--summary",
        action="store_true",
        help="통계 요약만 표시"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="색상 비활성화"
    )
    
    args = parser.parse_args()
    
    # 색상 비활성화 (파일로 리다이렉트하거나 --no-color 옵션 시)
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()
    
    # 뷰어 생성 및 로드
    viewer = APILogViewer(args.file)
    if not viewer.load():
        return
    
    # 필터링
    entries = viewer.filter(
        exclude_embedding=args.exclude_embedding,
        provider=args.provider,
        only_errors=args.errors,
        only_responses=args.responses,
        limit=args.limit
    )
    
    # 표시
    if args.summary:
        viewer.display_summary(entries)
    elif args.detailed:
        viewer.display_detailed(entries)
        viewer.display_summary(entries)
    else:
        viewer.display_compact(entries)
        if not args.limit:
            viewer.display_summary(entries)
    
    print()


if __name__ == "__main__":
    main()
