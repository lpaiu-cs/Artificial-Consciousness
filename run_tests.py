#!/usr/bin/env python
"""
CogBot Test Runner
간편하게 테스트를 실행하기 위한 스크립트
"""
import subprocess
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="CogBot 테스트 실행기")
    parser.add_argument(
        "--unit", "-u",
        action="store_true",
        help="유닛 테스트만 실행"
    )
    parser.add_argument(
        "--integration", "-i",
        action="store_true",
        help="통합 테스트만 실행"
    )
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="코드 커버리지 포함"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="상세 출력"
    )
    parser.add_argument(
        "--module", "-m",
        type=str,
        help="특정 모듈만 테스트 (예: stm_handler)"
    )
    parser.add_argument(
        "--fast", "-f",
        action="store_true",
        help="느린 테스트 제외"
    )
    parser.add_argument(
        "--smoke", "-s",
        action="store_true",
        help="추적된 smoke suite만 실행"
    )
    
    args = parser.parse_args()
    
    # 기본 pytest 명령
    cmd = ["python", "-m", "pytest"]
    
    # 옵션 추가
    if args.verbose:
        cmd.append("-v")
    
    if args.unit:
        cmd.extend(["-m", "unit"])
    elif args.integration:
        cmd.extend(["-m", "integration"])
    
    if args.fast:
        cmd.extend(["-m", "not slow"])
    
    if args.coverage:
        cmd.extend(["--cov=.", "--cov-report=term-missing", "--cov-report=html"])
    
    if args.module:
        cmd.append(f"tests/test_{args.module}.py")
    elif args.smoke:
        cmd.append("tests/test_smoke_suite.py")
    else:
        cmd.append("tests/")
    
    # 실행
    print(f"🧪 Running: {' '.join(cmd)}")
    print("=" * 60)
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Some tests failed (exit code: {result.returncode})")
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
