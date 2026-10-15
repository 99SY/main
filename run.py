"""VS Code에서 이 파일 하나만 실행(▷ Run Python File 또는 F5)하면 터미널에 바로 결과가 출력됩니다.

비교할 파일은 아래 OLD_STEP / NEW_STEP 경로만 바꾸면 됩니다.
(상대 경로는 이 파일이 있는 폴더 기준. 예제 파일이 없으면 자동 생성)
"""
import sys
from pathlib import Path

# ----------------------------------------------------------------------------
# 설정: 비교할 STEP 파일
OLD_STEP = "examples/old.step"
NEW_STEP = "examples/new.step"
SAVE_JSON = "out/report.json"  # None 이면 저장 안 함
SAVE_CSV = "out/report.csv"    # None 이면 저장 안 함
SHOW_DETAILS = True            # 객체별 상세(이동/회전/스케일 수치) 출력
# ----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # 어느 폴더에서 실행해도 step_diff 를 찾도록

from step_diff import Config, compare_step_files  # noqa: E402
from step_diff.report import format_details, format_text, write_csv, write_json  # noqa: E402


def _path(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def main():
    try:  # Windows 콘솔에서도 한글/기호가 깨지지 않도록
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    old, new = _path(OLD_STEP), _path(NEW_STEP)
    if not (old.exists() and new.exists()) and old.parent == ROOT / "examples":
        from step_diff.samples import write_demo

        print("예제 STEP 파일이 없어 새로 생성합니다 ...")
        write_demo(ROOT / "examples")
    for p in (old, new):
        if not p.exists():
            sys.exit(f"STEP 파일을 찾을 수 없습니다: {p}")

    print(f"OLD : {old}\nNEW : {new}\n")
    report = compare_step_files(old, new, Config())

    print("=" * 100)
    print(format_text(report))
    if SHOW_DETAILS:
        print("\n" + "=" * 100 + "\n상세 결과")
        print(format_details(report))
    print("\n" + "=" * 100)
    print(f"완료 ({report['summary']['elapsed_s']} s)")

    if SAVE_JSON:
        _path(SAVE_JSON).parent.mkdir(parents=True, exist_ok=True)
        write_json(report, _path(SAVE_JSON))
        print(f"JSON 저장: {_path(SAVE_JSON)}")
    if SAVE_CSV:
        _path(SAVE_CSV).parent.mkdir(parents=True, exist_ok=True)
        write_csv(report, _path(SAVE_CSV))
        print(f"CSV 저장 : {_path(SAVE_CSV)}")


if __name__ == "__main__":
    main()
