"""
model/data/splits/test.txt를 소스/occlusion 비율 기준으로 나눠 평가용 서브셋 목록을 만든다.

생성되는 서브셋 (model/data/splits/eval_subsets/):
- test_full.txt          : 전체 test set 그대로 (복사)
- test_public.txt         : 공개 데이터(coco/crowdhuman/widerperson)만
- test_synthetic.txt      : 합성(synthetic) 전체
- test_synthetic_occ30.txt / occ50.txt / occ70.txt : 합성 중 target_ratio 기준 버킷
- test_own_capture.txt   : 자체 촬영(own_capture) test subset (129장)

synthetic occlusion 버킷은 occlusion_augment.py가 이미지-변형(variant) 단위로 target_ratio를
동일하게 적용했으므로 (한 이미지 내 모든 박스가 같은 target_ratio) metadata.csv의
output_image -> target_ratio 매핑을 그대로 사용한다.

사용 예:
    python build_eval_subsets.py --test-txt model/data/splits/test.txt \
        --metadata model/data/synthetic/metadata.csv \
        --out model/data/splits/eval_subsets
"""
import argparse
import csv
from pathlib import Path


def load_target_ratio_map(metadata_csv):
    """output_image(파일명) -> target_ratio(float) 매핑."""
    mapping = {}
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["output_image"]] = float(row["target_ratio"])
    return mapping


def classify(path_str):
    p = path_str.replace("\\", "/")
    if "/processed/coco/" in p or "/processed/crowdhuman/" in p or "/processed/widerperson/" in p:
        return "public"
    if "/synthetic/" in p:
        return "synthetic"
    if "/own_capture/" in p:
        return "own_capture"
    return "other"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-txt", required=True)
    parser.add_argument("--metadata", required=True, help="model/data/synthetic/metadata.csv")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [l for l in Path(args.test_txt).read_text(encoding="utf-8").splitlines() if l.strip()]
    ratio_map = load_target_ratio_map(args.metadata)

    buckets = {
        "test_full": [],
        "test_public": [],
        "test_synthetic": [],
        "test_synthetic_occ30": [],
        "test_synthetic_occ50": [],
        "test_synthetic_occ70": [],
        "test_own_capture": [],
    }

    for line in lines:
        buckets["test_full"].append(line)
        kind = classify(line)
        filename = Path(line.replace("\\", "/")).name
        if kind == "public":
            buckets["test_public"].append(line)
        elif kind == "own_capture":
            buckets["test_own_capture"].append(line)
        elif kind == "synthetic":
            buckets["test_synthetic"].append(line)
            ratio = ratio_map.get(filename)
            if ratio is not None:
                bucket_key = f"test_synthetic_occ{int(round(ratio * 100))}"
                if bucket_key in buckets:
                    buckets[bucket_key].append(line)

    for name, items in buckets.items():
        out_path = out_dir / f"{name}.txt"
        out_path.write_text("\n".join(items) + ("\n" if items else ""), encoding="utf-8")
        print(f"{name}: {len(items)}장 -> {out_path}")


if __name__ == "__main__":
    main()
