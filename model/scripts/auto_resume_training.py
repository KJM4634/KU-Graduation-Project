"""
예기치 않은 재부팅 등으로 학습이 중단됐을 경우, 로그인 시 자동으로 이어서 학습을 재개한다
(Windows 작업 스케줄러 "로그온 시" 트리거로 등록해 사용).

판단 기준: results.csv의 마지막 수정 시각이 STALE_MINUTES보다 오래됐고, 동시에 해당 학습을
실행하는 python 프로세스가 현재 없다면 "중단된 것"으로 보고 last.pt에서 재개한다.
이미 정상 실행 중이면 아무 것도 하지 않는다 (중복 실행 방지).

사용 예 (작업 스케줄러에 등록):
    pythonw.exe model/scripts/auto_resume_training.py --run-dir model/experiments/train/n_full
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

STALE_MINUTES = 20


def is_stale(results_csv, stale_minutes):
    if not results_csv.exists():
        return True
    age_sec = time.time() - results_csv.stat().st_mtime
    return age_sec > stale_minutes * 60


def training_process_running(run_dir_name):
    """같은 run 이름을 인자로 가진 python 프로세스가 있는지 확인 (Windows wmic 사용)."""
    try:
        out = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'", "get", "CommandLine"],
            capture_output=True, text=True, timeout=15,
        )
        return run_dir_name in out.stdout
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="예: model/experiments/train/n_full")
    parser.add_argument("--stale-minutes", type=int, default=STALE_MINUTES)
    parser.add_argument("--log", default=None, help="이 스크립트 자체의 실행 로그 파일")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    results_csv = run_dir / "results.csv"
    last_pt = run_dir / "weights" / "last.pt"
    log_path = Path(args.log) if args.log else run_dir.parent / f"{run_dir.name}_autoresume.log"

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    if not is_stale(results_csv, args.stale_minutes):
        log("정상 진행 중으로 판단, 아무 것도 하지 않음 (results.csv 최근 갱신됨)")
        return

    if training_process_running(run_dir.name):
        log("results.csv는 오래됐지만 학습 프로세스가 이미 실행 중, 대기")
        return

    if not last_pt.exists():
        log(f"경고: {last_pt} 없음, 재개 불가")
        return

    log(f"학습이 중단된 것으로 판단 -> {last_pt} 에서 재개 시작")
    project_root = Path(__file__).resolve().parents[2]
    python_exe = project_root / ".venv" / "Scripts" / "python.exe"
    console_log = run_dir.parent / f"{run_dir.name}_console.log"

    with open(console_log, "a", encoding="utf-8") as out_f:
        subprocess.Popen(
            [str(python_exe), "-c",
             f"from ultralytics import YOLO; YOLO(r'{last_pt}').train(resume=True)"],
            stdout=out_f, stderr=out_f, cwd=str(project_root),
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    log("재개 프로세스 실행함")


if __name__ == "__main__":
    main()
