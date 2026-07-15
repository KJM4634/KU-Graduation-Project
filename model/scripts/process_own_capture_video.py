"""
자체 촬영 영상에서 추출한 프레임을, "완전히 서있거나 걷는 중"인 프레임과
"누워있는(가려짐 학습 대상) 중"인 프레임으로 분류해 각각 별도 폴더로 나눈다.

가려짐 비율 자체는 자동으로 추정하지 않는다 (픽셀 diff 기반 자동 측정을 시도했으나
영상 압축/조명 변화로 인한 노이즈가 실제 가려짐 신호보다 크거나 비슷한 수준이라
신뢰할 수 없어 폐기함 — docs/DATASET.md 참고). 라벨링(bbox)은 사람이 프레임을
직접 보면서 그린다.

분류 방법: YOLO11로 검출한 person bbox의 종횡비(width/height)를 사용한다.
- 서있는/걷는 사람은 bbox가 세로로 길다 (종횡비가 작음)
- 누운 사람은 bbox가 가로로 길다 (종횡비가 큼)
프레임 단위 노이즈를 줄이기 위해 시간축 median filter로 스무딩한다.

사용 예:
    python process_own_capture_video.py \
        --frames model/data/own_capture/extracted_frames \
        --out model/data/own_capture \
        --model model/weights/yolo11n.pt
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

STANDING_ASPECT_THRESHOLD = 0.75  # w/h 이 값보다 작으면 "서있음"(세로로 긴 bbox)
SMOOTH_WINDOW = 9  # 시간축 median filter 창 크기 (8fps 기준 약 1.1초)


def detect_all_frames(model, frame_paths, conf=0.1):
    """각 프레임에서 가장 넓은 person 검출 결과(box, conf)를 반환."""
    records = []
    results = model.predict([str(p) for p in frame_paths], classes=[0], conf=conf, verbose=False, stream=True)
    for path, res in zip(frame_paths, results):
        if len(res.boxes) == 0:
            records.append({"path": path, "box": None, "conf": 0.0})
            continue
        areas = (res.boxes.xyxy[:, 2] - res.boxes.xyxy[:, 0]) * (res.boxes.xyxy[:, 3] - res.boxes.xyxy[:, 1])
        best = int(areas.argmax())
        box = res.boxes.xyxy[best].tolist()
        conf_val = float(res.boxes.conf[best])
        records.append({"path": path, "box": box, "conf": conf_val})
    return records


def classify_standing_lying(records):
    """bbox 종횡비 기반 1차 분류 + median filter 스무딩. 반환: ['standing'|'lying', ...]"""
    raw = []
    last_label = "lying"
    for r in records:
        if r["box"] is None:
            raw.append(last_label)  # 미검출 프레임은 직전 라벨 유지 (스무딩으로 재보정됨)
            continue
        x1, y1, x2, y2 = r["box"]
        w, h = x2 - x1, y2 - y1
        aspect = w / h if h > 0 else 1.0
        label = "standing" if aspect < STANDING_ASPECT_THRESHOLD else "lying"
        raw.append(label)
        last_label = label

    smoothed = []
    n = len(raw)
    half = SMOOTH_WINDOW // 2
    for i in range(n):
        window = raw[max(0, i - half): min(n, i + half + 1)]
        lying_count = window.count("lying")
        smoothed.append("lying" if lying_count >= len(window) / 2 else "standing")
    return smoothed


def process_video(video_stem, frames_dir, out_root, model):
    frame_dir = frames_dir / video_stem
    frame_paths = sorted(frame_dir.glob("frame_*.jpg"))
    if not frame_paths:
        print(f"[{video_stem}] 프레임 없음, 건너뜀")
        return 0, 0

    records = detect_all_frames(model, frame_paths)
    labels = classify_standing_lying(records)

    standing_dir = out_root / "frames_standing" / video_stem
    lying_dir = out_root / "frames_lying" / video_stem
    standing_dir.mkdir(parents=True, exist_ok=True)
    lying_dir.mkdir(parents=True, exist_ok=True)

    n_standing = n_lying = 0
    for r, label in zip(records, labels):
        dest_dir = standing_dir if label == "standing" else lying_dir
        shutil.copy2(r["path"], dest_dir / r["path"].name)
        if label == "standing":
            n_standing += 1
        else:
            n_lying += 1

    print(f"[{video_stem}] 총 {len(records)}프레임 -> lying(라벨링 대상) {n_lying}, standing/walking(제외) {n_standing}")
    return n_lying, n_standing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", required=True, help="extract_frames.py 출력 디렉토리 (영상별 하위 폴더)")
    parser.add_argument("--out", required=True, help="model/data/own_capture 등 출력 루트")
    parser.add_argument("--model", default="model/weights/yolo11n.pt")
    args = parser.parse_args()

    frames_dir = Path(args.frames)
    out_root = Path(args.out)
    model = YOLO(args.model)

    video_stems = sorted(p.name for p in frames_dir.iterdir() if p.is_dir())
    total_lying, total_standing = 0, 0
    for stem in video_stems:
        n_lying, n_standing = process_video(stem, frames_dir, out_root, model)
        total_lying += n_lying
        total_standing += n_standing

    print(f"\n=== 전체 결과 ===")
    print(f"lying(라벨링 대상) 프레임: {total_lying}")
    print(f"standing/walking(제외) 프레임: {total_standing}")


if __name__ == "__main__":
    main()
