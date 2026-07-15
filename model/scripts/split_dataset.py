"""
YOLO 포맷으로 변환된 소스별 데이터셋을 train/val/test로 분할한다 (PRD 7.4절 비율 기준).

| 소스 | train | val | test |
|---|---|---|---|
| 공개+occlusion 서브셋 (coco/crowdhuman/widerperson) | 80% | 10% | 10% |
| OCHuman (bbox가 full-extent인지 불확실 → 검증 전용) | 0% | 50% | 50% |
| 합성(synthetic) | 80% | 10% | 10% |
| 자체 촬영(own_capture) | 30% | 20% | 50% |

출력: model/data/splits/{train,val,test}.txt (각 줄에 이미지 절대경로 1개).
라벨 경로는 YOLO 관례대로 "/images/"를 "/labels/"로, 확장자를 ".txt"로 바꿔 추론한다.

사용 예:
    python split_dataset.py --data-root model/data --seed 42
"""
import argparse
import random
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# (source 이름, images 디렉토리, ratios(train, val, test)) — data-root 기준 상대경로
DEFAULT_SOURCES = [
    ("coco", "processed/coco/images", (0.8, 0.1, 0.1)),
    ("crowdhuman", "processed/crowdhuman/images", (0.8, 0.1, 0.1)),
    ("widerperson", "processed/widerperson/images", (0.8, 0.1, 0.1)),
    ("ochuman", "processed/ochuman/images", (0.0, 0.5, 0.5)),  # 검증 전용 (docs/DATASET.md 참고)
    ("synthetic", "synthetic/images", (0.8, 0.1, 0.1)),
    ("own_capture", "own_capture/labeled/images", (0.3, 0.2, 0.5)),
]


def collect_labeled_images(images_dir):
    """이미지 중 대응하는 라벨(.txt) 파일이 존재하는 것만 수집 (빈 라벨=배경 이미지도 포함)."""
    images_dir = Path(images_dir)
    if not images_dir.exists():
        return []
    labels_dir = images_dir.parent / "labels"
    items = []
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            items.append(img_path.resolve())
    return items


def split_list(items, ratios, rng):
    """items를 ratios(train,val,test) 비율로 분할. 소수점 반올림은 test 쪽에 몰아 처리."""
    items = items[:]
    rng.shuffle(items)
    n = len(items)
    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="model/data", help="model/data 디렉토리 경로")
    parser.add_argument("--out", default=None, help="분할 결과 출력 디렉토리 (기본: <data-root>/splits)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out) if args.out else data_root / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    all_train, all_val, all_test = [], [], []
    for name, rel_images_dir, ratios in DEFAULT_SOURCES:
        items = collect_labeled_images(data_root / rel_images_dir)
        if not items:
            print(f"[{name}] 라벨링된 이미지 없음 (건너뜀): {data_root / rel_images_dir}")
            continue
        train, val, test = split_list(items, ratios, rng)
        all_train += train
        all_val += val
        all_test += test
        print(f"[{name}] 총 {len(items)}장 -> train {len(train)} / val {len(val)} / test {len(test)}")

    for split_name, split_items in (("train", all_train), ("val", all_val), ("test", all_test)):
        out_path = out_dir / f"{split_name}.txt"
        out_path.write_text("\n".join(str(p) for p in split_items) + ("\n" if split_items else ""),
                             encoding="utf-8")

    print(f"\n최종: train {len(all_train)} / val {len(all_val)} / test {len(all_test)}")
    print(f"분할 목록 저장 위치: {out_dir}")


if __name__ == "__main__":
    main()
