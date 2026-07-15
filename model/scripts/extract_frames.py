"""
자체 촬영 영상(고정 카메라, 눕기->덮이기->일어나기 연속 촬영)에서 프레임을 촘촘하게 추출한다.

원본 fps와 무관하게 목표 fps(기본 8장/초)에 맞춰 프레임을 골라 저장하며,
각 프레임의 원본 영상 내 타임스탬프를 manifest.csv에 기록한다.

사용 예:
    python extract_frames.py \
        --videos model/data/own_capture/raw_videos \
        --out model/data/own_capture/frames_all \
        --fps 8
"""
import argparse
import csv
from pathlib import Path

import cv2


def extract_video(video_path, out_dir, target_fps):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"경고: {video_path.name} 열기 실패, 건너뜀")
        return []

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / src_fps if src_fps else 0

    video_out_dir = out_dir / video_path.stem
    video_out_dir.mkdir(parents=True, exist_ok=True)

    # 목표 fps에 맞춰 추출할 타임스탬프 목록 생성 후, 각 타임스탬프에 가장 가까운 원본 프레임을 선택
    n_target = int(duration * target_fps)
    manifest_rows = []
    frame_idx = 0
    next_target_i = 0
    ret = True
    while ret:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / src_fps
        target_t = next_target_i / target_fps
        if t >= target_t and next_target_i < n_target + 1:
            out_name = f"frame_{next_target_i:06d}.jpg"
            cv2.imwrite(str(video_out_dir / out_name), frame)
            manifest_rows.append([video_path.stem, out_name, next_target_i, f"{t:.3f}"])
            next_target_i += 1
        frame_idx += 1
    cap.release()
    print(f"{video_path.name}: 원본 {n_frames}프레임({src_fps:.1f}fps, {duration:.1f}초) "
          f"-> 추출 {len(manifest_rows)}프레임 ({target_fps}fps 목표)")
    return manifest_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", required=True, help="원본 영상(.mp4) 디렉토리")
    parser.add_argument("--out", required=True, help="추출된 프레임 출력 디렉토리 (영상별 하위 폴더 생성)")
    parser.add_argument("--fps", type=float, default=8.0, help="추출 목표 fps (기본 8)")
    args = parser.parse_args()

    videos_dir = Path(args.videos)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(videos_dir.glob("*.mp4"))
    if not video_paths:
        print(f"경고: {videos_dir} 에서 영상을 찾지 못함")
        return

    manifest_path = out_dir / "manifest.csv"
    all_rows = []
    for video_path in video_paths:
        all_rows.extend(extract_video(video_path, out_dir, args.fps))

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "frame_file", "frame_index", "timestamp_sec"])
        writer.writerows(all_rows)

    print(f"\n총 {len(all_rows)}프레임 추출 완료. manifest: {manifest_path}")


if __name__ == "__main__":
    main()
