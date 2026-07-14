"""
CrowdHuman의 vbox(visible)/fbox(full) 비율로 occlusion 정도를 분석한다.

occlusion_ratio = 1 - (vbox_area / fbox_area)  (0=완전히 보임, 1=완전히 가려짐)

박스 단위와 이미지 단위(이미지 내 최대 occlusion 기준) 두 가지로 30/50/70% 구간별
개수를 집계해, PRD 7장 occlusion 비율 구간과 맞춰 서브셋 규모를 가늠할 수 있게 한다.
--min-occlusion 옵션을 주면 해당 비율 이상인 박스가 하나라도 있는 이미지 ID 목록을
파일로 저장한다 (서브셋 구성용).

사용 예:
    python analyze_crowdhuman_occlusion.py \
        --odgt model/data/raw/crowdhuman/annotation_train.odgt \
             model/data/raw/crowdhuman/annotation_val.odgt \
        --min-occlusion 0.5 --out-subset model/data/processed/crowdhuman/occ50_subset.txt
"""
import argparse
import json
from pathlib import Path


def box_area(box):
    _, _, w, h = box
    return max(0.0, w) * max(0.0, h)


def compute_occlusion_ratios(odgt_paths):
    """반환: [(image_id, [box별 occlusion_ratio, ...]), ...]"""
    results = []
    for odgt_path in odgt_paths:
        with open(odgt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                ratios = []
                for gt in record.get("gtboxes", []):
                    if gt.get("tag") != "person":
                        continue
                    if gt.get("extra", {}).get("ignore", 0) == 1:
                        continue
                    fbox_area = box_area(gt["fbox"])
                    vbox_area = box_area(gt["vbox"])
                    if fbox_area <= 0:
                        continue
                    ratio = 1.0 - (vbox_area / fbox_area)
                    ratios.append(max(0.0, min(1.0, ratio)))
                if ratios:
                    results.append((record["ID"], ratios))
    return results


def bucket_label(ratio):
    if ratio < 0.3:
        return "<30%"
    elif ratio < 0.5:
        return "30~50%"
    elif ratio < 0.7:
        return "50~70%"
    else:
        return ">=70%"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--odgt", nargs="+", required=True, help="odgt 파일 경로 (여러 개 가능)")
    parser.add_argument("--min-occlusion", type=float, default=None,
                         help="지정 시, 박스 occlusion_ratio가 이 값 이상인 이미지 ID 목록을 저장")
    parser.add_argument("--out-subset", default=None, help="--min-occlusion 서브셋 이미지 ID 저장 경로")
    args = parser.parse_args()

    data = compute_occlusion_ratios(args.odgt)
    n_images = len(data)
    n_boxes = sum(len(ratios) for _, ratios in data)

    box_buckets = {"<30%": 0, "30~50%": 0, "50~70%": 0, ">=70%": 0}
    image_buckets = {"<30%": 0, "30~50%": 0, "50~70%": 0, ">=70%": 0}

    subset_ids = []
    for image_id, ratios in data:
        for r in ratios:
            box_buckets[bucket_label(r)] += 1
        max_ratio = max(ratios)
        image_buckets[bucket_label(max_ratio)] += 1
        if args.min_occlusion is not None and max_ratio >= args.min_occlusion:
            subset_ids.append(image_id)

    print(f"총 이미지: {n_images}, 총 person 박스: {n_boxes}")
    print("\n[박스 단위 occlusion 분포]")
    for k, v in box_buckets.items():
        print(f"  {k}: {v}개 ({v/n_boxes*100:.1f}%)")
    print("\n[이미지 단위 occlusion 분포] (이미지 내 최대 occlusion 박스 기준)")
    for k, v in image_buckets.items():
        print(f"  {k}: {v}장 ({v/n_images*100:.1f}%)")

    if args.min_occlusion is not None:
        print(f"\nocclusion >= {args.min_occlusion:.0%} 박스를 포함하는 이미지: {len(subset_ids)}장")
        if args.out_subset:
            Path(args.out_subset).write_text("\n".join(subset_ids) + "\n")
            print(f"이미지 ID 목록 저장: {args.out_subset}")


if __name__ == "__main__":
    main()
