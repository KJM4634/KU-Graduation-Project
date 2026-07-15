"""
frames_lying/ 프레임이 너무 많고(8fps 추출이라 인접 프레임끼리 거의 동일) 중복도 심해,
"시각적으로 충분히 달라지는 지점"만 남기는 방식으로 서브샘플링한다.

방법: 각 영상의 프레임을 시간 순서대로 훑으며, 직전에 채택(keep)한 프레임과
비교해 내용이 충분히 달라졌을 때만 새로 채택하는 그리디(greedy) 방식.
- 비교는 그레이스케일로 변환 후 작은 크기(기본 32x32)로 다운샘플링한 뒤 평균 절대
  차이(MAD)를 사용 — 압축 노이즈/미세한 조명 변화에 덜 민감하면서도 자세 변화나
  이불이 덮이는 큰 변화는 잘 잡아냄 (정밀한 occlusion 비율 추정과 달리, 여기서는
  "많이 다른지"만 판단하면 되므로 이 정도로도 충분함).
- min-gap: 노이즈로 인해 너무 자주 채택되는 것을 방지하는 최소 프레임 간격
- max-gap: 변화가 거의 없는 구간(예: 완전히 덮인 채 정지)에서도 일정 간격으로는
  대표 프레임을 남기기 위한 최대 프레임 간격

사용 예:
    python subsample_frames.py \
        --frames model/data/own_capture/frames_lying \
        --out model/data/own_capture/subsampled_frames \
        --threshold 12 --min-gap 4 --max-gap 24
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def frame_signature(path, size=32):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)


def subsample_video(video_stem, frames_dir, out_dir, threshold, min_gap, max_gap):
    frame_paths = sorted((frames_dir / video_stem).glob("frame_*.jpg"))
    if not frame_paths:
        return 0, 0

    out_video_dir = out_dir / video_stem
    out_video_dir.mkdir(parents=True, exist_ok=True)

    kept_indices = [0]
    last_sig = frame_signature(frame_paths[0])
    last_kept_i = 0

    for i in range(1, len(frame_paths)):
        gap = i - last_kept_i
        sig = frame_signature(frame_paths[i])
        diff = float(np.abs(sig - last_sig).mean())

        if (diff >= threshold and gap >= min_gap) or gap >= max_gap:
            kept_indices.append(i)
            last_sig = sig
            last_kept_i = i

    for i in kept_indices:
        shutil.copy2(frame_paths[i], out_video_dir / frame_paths[i].name)

    return len(frame_paths), len(kept_indices)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", required=True, help="frames_lying 등 입력 디렉토리 (영상별 하위 폴더)")
    parser.add_argument("--out", required=True, help="서브샘플링 결과 출력 디렉토리")
    parser.add_argument("--threshold", type=float, default=12.0, help="그레이스케일 32x32 기준 평균 절대차 임계값")
    parser.add_argument("--min-gap", type=int, default=4, help="최소 프레임 간격 (8fps 기준 4=0.5초)")
    parser.add_argument("--max-gap", type=int, default=24, help="최대 프레임 간격 (8fps 기준 24=3초)")
    args = parser.parse_args()

    frames_dir = Path(args.frames)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_stems = sorted(p.name for p in frames_dir.iterdir() if p.is_dir())
    total_in, total_out = 0, 0
    for stem in video_stems:
        n_in, n_out = subsample_video(stem, frames_dir, out_dir, args.threshold, args.min_gap, args.max_gap)
        total_in += n_in
        total_out += n_out
        print(f"[{stem}] {n_in} -> {n_out}")

    print(f"\n총 {total_in}장 -> {total_out}장으로 서브샘플링 (threshold={args.threshold}, "
          f"min_gap={args.min_gap}, max_gap={args.max_gap})")


if __name__ == "__main__":
    main()
