"""
COCO train2017/val2017 어노테이션에서 person 카테고리가 포함된 이미지만 골라
개별 URL로 다운로드한다 (전체 zip을 받지 않아 대역폭 절약, docs/DATASET.md 2.2절).

사전 준비: model/data/raw/coco/annotations/ 에 instances_train2017.json,
instances_val2017.json 이 있어야 함 (annotations_trainval2017.zip 압축 해제 후 배치).

사용 예:
    python download_coco_person_subset.py --out model/data/raw/coco/images --workers 32
"""
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_URL = "http://images.cocodataset.org"


def collect_person_images(annotations_dir):
    """(split, file_name) 튜플 리스트 반환. split은 'train2017' 또는 'val2017'."""
    items = []
    for split in ("val2017", "train2017"):
        ann_path = Path(annotations_dir) / f"instances_{split}.json"
        if not ann_path.exists():
            print(f"경고: {ann_path} 없음, 건너뜀")
            continue
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        person_cat_ids = {c["id"] for c in data["categories"] if c["name"] == "person"}
        img_ids_with_person = {a["image_id"] for a in data["annotations"] if a["category_id"] in person_cat_ids}
        images_by_id = {img["id"]: img for img in data["images"]}
        for image_id in img_ids_with_person:
            img = images_by_id.get(image_id)
            if img:
                items.append((split, img["file_name"]))
    return items


DOWNLOAD_TIMEOUT_SEC = 20  # 타임아웃 미설정 시 죽은 연결에서 무한 대기할 수 있어 명시적으로 지정


def download_one(split, file_name, out_dir):
    dst = out_dir / file_name
    if dst.exists() and dst.stat().st_size > 0:
        return "skipped"
    url = f"{BASE_URL}/{split}/{file_name}"
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SEC) as resp, open(dst, "wb") as f:
            f.write(resp.read())
        return "downloaded"
    except Exception as e:
        return f"error:{e}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="model/data/raw/coco/annotations")
    parser.add_argument("--out", default="model/data/raw/coco/images")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="테스트용 상한 (지정 시 앞에서부터 N장만)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = collect_person_images(args.annotations)
    print(f"person 포함 이미지 총 {len(items)}장 확인")
    if args.limit:
        items = items[: args.limit]
        print(f"--limit 적용: {len(items)}장만 다운로드")

    start = time.time()
    counts = {"downloaded": 0, "skipped": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_one, split, fn, out_dir): fn for split, fn in items}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result.startswith("error"):
                counts["error"] += 1
                print(f"실패: {futures[future]} ({result})")
            elif result == "skipped":
                counts["skipped"] += 1
            else:
                counts["downloaded"] += 1
            if i % 500 == 0 or i == len(items):
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0
                print(f"진행: {i}/{len(items)} (다운로드 {counts['downloaded']}, 스킵 {counts['skipped']}, "
                      f"실패 {counts['error']}) - {rate:.1f}장/초, 경과 {elapsed:.0f}초")

    print(f"완료: {counts}")


if __name__ == "__main__":
    main()
