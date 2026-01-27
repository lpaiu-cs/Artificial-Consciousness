#!/usr/bin/env python
"""
CogBot API Log Viewer
로그를 깔끔하게 정리해서 보여주는 CLI 도구
"""
import json
import argparse
import os
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
    
    def display_compact(self, entries: List[Dict]):
        """컴팩트 뷰 (한 줄씩)"""
        print(f"\n{Colors.BOLD}{'시간':<10} {'타입':<12} {'Provider':<8} {'모델':<20} {'상태':<8} {'소요시간':<10}{Colors.END}")
        print("─" * 75)
        
        for e in entries:
            time_str = self.format_timestamp(e.get('timestamp', ''))
            entry_type = e.get('type', 'UNKNOWN')
            provider = e.get('provider', '-')
            model = e.get('model', '-')[:18]
            
            # 상태 색상
            if 'REQUEST' in entry_type:
                status = f"{Colors.BLUE}→ REQ{Colors.END}"
                duration = "-"
            else:
                success = e.get('success', True)
                if success:
                    status = f"{Colors.GREEN}✓ OK{Colors.END}"
                else:
                    status = f"{Colors.RED}✗ ERR{Colors.END}"
                duration = self.format_duration(e.get('duration_ms', 0))
            
            # 타입 색상
            if 'EMBEDDING' in entry_type:
                type_str = f"{Colors.GRAY}EMBED{Colors.END}"
            elif 'CHAT' in entry_type:
                type_str = f"{Colors.CYAN}CHAT{Colors.END}"
            else:
                type_str = entry_type[:10]
            
            print(f"{time_str:<10} {type_str:<20} {provider:<8} {model:<20} {status:<16} {duration:<10}")
    
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
            
            print(f"\n{header_color}{'━' * 60}{Colors.END}")
            print(f"{header_color}[{i}] {entry_type}{Colors.END}")
            print(f"{Colors.GRAY}시간: {e.get('timestamp', 'N/A')}{Colors.END}")
            
            # 공통 정보
            if 'provider' in e:
                print(f"  Provider: {Colors.BOLD}{e['provider']}{Colors.END}")
            if 'model' in e:
                print(f"  모델: {e['model']}")
            
            # Request 정보
            if 'REQUEST' in entry_type:
                if 'system_prompt' in e:
                    print(f"  {Colors.GRAY}[System]{Colors.END} {e['system_prompt'][:100]}...")
                if 'user_prompt' in e:
                    print(f"  {Colors.CYAN}[User]{Colors.END} {e['user_prompt'][:100]}...")
                if e.get('json_mode'):
                    print(f"  {Colors.YELLOW}JSON Mode: ON{Colors.END}")
            
            # Response 정보
            if 'RESPONSE' in entry_type:
                duration = self.format_duration(e.get('duration_ms', 0))
                print(f"  소요시간: {Colors.BOLD}{duration}{Colors.END}")
                
                if e.get('success'):
                    if 'response' in e:
                        resp = str(e['response'])[:150]
                        print(f"  {Colors.GREEN}[Response]{Colors.END} {resp}...")
                    if 'token_usage' in e:
                        t = e['token_usage']
                        print(f"  토큰: {t.get('prompt_tokens', 0)} + {t.get('completion_tokens', 0)} = {Colors.BOLD}{t.get('total_tokens', 0)}{Colors.END}")
                else:
                    print(f"  {Colors.RED}❌ 에러: {e.get('error', 'Unknown error')}{Colors.END}")
    
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
    
    # 색상 비활성화
    if args.no_color:
        for attr in dir(Colors):
            if not attr.startswith('_'):
                setattr(Colors, attr, '')
    
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
