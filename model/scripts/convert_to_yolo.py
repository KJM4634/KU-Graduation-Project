"""
공개 데이터셋(COCO/CrowdHuman/WiderPerson/OCHuman)을 YOLO11 라벨 포맷으로 통일 변환.

모든 소스는 단일 클래스 "person"(class 0)으로 매핑하며, full-extent bbox
(가려짐과 무관하게 사람의 실제 전체 범위)를 라벨로 사용한다 (PRD 6장/8장).

소스별 처리 방침:
- COCO        : instances_*.json 의 category_id=1(person) bbox 사용
- CrowdHuman  : annotation_*.odgt 의 gtboxes 중 tag="person"인 항목의 fbox(전체범위) 사용,
                tag="mask"(무시 대상)는 제외
- WiderPerson : 클래스 1(pedestrians)/2(riders)/3(partially-visible persons)만 person으로 매핑.
                4(ignore regions)/5(crowd)는 개별 full-extent person 라벨로 보기 어려워 제외
                (docs/DATASET.md 참고)
- OCHuman     : COCO-style json의 bbox 사용. bbox가 amodal(full-extent)인지 원 논문 기준으로
                확정되지 않았으므로, 학습보다는 검증(val/test) 용도로 사용할 것을 권장

각 변환 함수는 (images_dir, out_dir) 에 YOLO 포맷 images/labels 를 생성한다.
원본 데이터가 없어도 스크립트 자체는 이 파일 하단 __main__ 블록에서
CLI로 개별 소스를 선택해 실행할 수 있다 (데이터는 raw/ 아래 문서화된 경로에 위치해야 함).
"""
import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

CLASS_PERSON = 0


def _ensure_out_dirs(out_dir):
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    return out_dir


def _copy_image(src_path, out_dir, mode="copy"):
    dst = out_dir / "images" / src_path.name
    if mode == "move":
        shutil.move(str(src_path), dst)
    elif mode == "symlink":
        if not dst.exists():
            dst.symlink_to(src_path.resolve())
    else:
        shutil.copy2(src_path, dst)
    return dst


def _write_yolo_label(out_dir, stem, boxes_xyxy, img_w, img_h):
    """boxes_xyxy: [(x1,y1,x2,y2), ...] 픽셀 좌표. 클래스는 항상 person(0)."""
    lines = []
    for x1, y1, x2, y2 in boxes_xyxy:
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(img_w), x2), min(float(img_h), y2)
        if x2 <= x1 or y2 <= y1:
            continue
        xc = ((x1 + x2) / 2) / img_w
        yc = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f"{CLASS_PERSON} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    (out_dir / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def convert_coco(annotations_json, images_dir, out_dir, mode="copy"):
    """COCO instances_*.json → YOLO. category_id=1(person)만 추출."""
    images_dir, out_dir = Path(images_dir), _ensure_out_dirs(out_dir)
    coco = json.loads(Path(annotations_json).read_text(encoding="utf-8"))

    person_cat_ids = {c["id"] for c in coco["categories"] if c["name"] == "person"}
    images_by_id = {img["id"]: img for img in coco["images"]}

    boxes_by_image = {}
    for ann in coco["annotations"]:
        if ann["category_id"] not in person_cat_ids:
            continue
        x, y, w, h = ann["bbox"]  # COCO: [x,y,width,height], 좌상단 기준
        boxes_by_image.setdefault(ann["image_id"], []).append((x, y, x + w, y + h))

    converted = 0
    for image_id, boxes in boxes_by_image.items():
        img_meta = images_by_id.get(image_id)
        if img_meta is None:
            continue
        src_path = images_dir / img_meta["file_name"]
        if not src_path.exists():
            continue
        _copy_image(src_path, out_dir, mode)
        _write_yolo_label(out_dir, src_path.stem, boxes, img_meta["width"], img_meta["height"])
        converted += 1

    print(f"[COCO] 변환 완료: 이미지 {converted}장")
    return converted


def convert_crowdhuman(odgt_path, images_dir, out_dir, mode="copy"):
    """CrowdHuman .odgt (한 줄당 JSON) → YOLO. tag="person"의 fbox(전체범위) 사용."""
    images_dir, out_dir = Path(images_dir), _ensure_out_dirs(out_dir)

    converted = 0
    with open(odgt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image_id = record["ID"]
            src_path = images_dir / f"{image_id}.jpg"
            if not src_path.exists():
                continue

            boxes = []
            for gt in record.get("gtboxes", []):
                if gt.get("tag") != "person":
                    continue  # "mask" 등 무시 대상 제외
                if gt.get("extra", {}).get("ignore", 0) == 1:
                    continue
                fx, fy, fw, fh = gt["fbox"]  # full box (전체범위, occlusion 포함)
                boxes.append((fx, fy, fx + fw, fy + fh))

            if not boxes:
                continue
            with Image.open(src_path) as im:
                img_w, img_h = im.size
            _copy_image(src_path, out_dir, mode)
            _write_yolo_label(out_dir, src_path.stem, boxes, img_w, img_h)
            converted += 1

    print(f"[CrowdHuman] 변환 완료: 이미지 {converted}장")
    return converted


# WiderPerson 클래스 정의 (공식 문서 기준)
WIDERPERSON_PERSON_CLASSES = {1, 2, 3}  # pedestrians, riders, partially-visible persons
WIDERPERSON_EXCLUDED_CLASSES = {4, 5}   # ignore regions, crowd — 개별 full-extent 라벨로 부적합


def convert_widerperson(annotations_dir, images_dir, out_dir, mode="copy"):
    """WiderPerson {image}.jpg.txt → YOLO. 클래스 1/2/3만 person으로 매핑, 4/5는 제외."""
    annotations_dir, images_dir, out_dir = Path(annotations_dir), Path(images_dir), _ensure_out_dirs(out_dir)

    converted = 0
    for ann_path in sorted(annotations_dir.glob("*.jpg.txt")):
        image_name = ann_path.name[: -len(".txt")]
        src_path = images_dir / image_name
        if not src_path.exists():
            continue

        lines = ann_path.read_text().strip().splitlines()
        boxes = []
        for line in lines[1:] if lines and lines[0].strip().isdigit() else lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, x1, y1, x2, y2 = (int(parts[0]), *map(float, parts[1:]))
            if cls not in WIDERPERSON_PERSON_CLASSES:
                continue
            boxes.append((x1, y1, x2, y2))

        if not boxes:
            continue
        with Image.open(src_path) as im:
            img_w, img_h = im.size
        _copy_image(src_path, out_dir, mode)
        _write_yolo_label(out_dir, src_path.stem, boxes, img_w, img_h)
        converted += 1

    print(f"[WiderPerson] 변환 완료: 이미지 {converted}장 (ignore/crowd 클래스는 제외)")
    return converted


def convert_ochuman(annotations_json, images_dir, out_dir, mode="copy"):
    """OCHuman COCO-style json → YOLO.

    주의: bbox가 amodal(full-extent)인지 원 논문 기준 확정되지 않음 (docs/DATASET.md 참고).
    기본적으로 val/test(검증) 용도로만 사용을 권장하며, 학습에 포함하려면 별도 검토 필요.
    """
    images_dir, out_dir = Path(images_dir), _ensure_out_dirs(out_dir)
    coco = json.loads(Path(annotations_json).read_text(encoding="utf-8"))
    images_by_id = {img["id"]: img for img in coco["images"]}

    boxes_by_image = {}
    for ann in coco["annotations"]:
        if "bbox" not in ann or ann["bbox"] is None:
            continue
        x, y, w, h = ann["bbox"]
        boxes_by_image.setdefault(ann["image_id"], []).append((x, y, x + w, y + h))

    converted = 0
    for image_id, boxes in boxes_by_image.items():
        img_meta = images_by_id.get(image_id)
        if img_meta is None:
            continue
        src_path = images_dir / img_meta["file_name"]
        if not src_path.exists():
            continue
        _copy_image(src_path, out_dir, mode)
        _write_yolo_label(out_dir, src_path.stem, boxes, img_meta["width"], img_meta["height"])
        converted += 1

    print(f"[OCHuman] 변환 완료: 이미지 {converted}장 (검증 용도 권장, docs/DATASET.md 참고)")
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="source", required=True)

    p_coco = sub.add_parser("coco")
    p_coco.add_argument("--annotations", required=True)
    p_coco.add_argument("--images", required=True)
    p_coco.add_argument("--out", required=True)

    p_ch = sub.add_parser("crowdhuman")
    p_ch.add_argument("--odgt", required=True)
    p_ch.add_argument("--images", required=True)
    p_ch.add_argument("--out", required=True)

    p_wp = sub.add_parser("widerperson")
    p_wp.add_argument("--annotations", required=True)
    p_wp.add_argument("--images", required=True)
    p_wp.add_argument("--out", required=True)

    p_oc = sub.add_parser("ochuman")
    p_oc.add_argument("--annotations", required=True)
    p_oc.add_argument("--images", required=True)
    p_oc.add_argument("--out", required=True)

    for sp in (p_coco, p_ch, p_wp, p_oc):
        sp.add_argument("--mode", choices=["copy", "move", "symlink"], default="copy")

    args = parser.parse_args()

    if args.source == "coco":
        convert_coco(args.annotations, args.images, args.out, args.mode)
    elif args.source == "crowdhuman":
        convert_crowdhuman(args.odgt, args.images, args.out, args.mode)
    elif args.source == "widerperson":
        convert_widerperson(args.annotations, args.images, args.out, args.mode)
    elif args.source == "ochuman":
        convert_ochuman(args.annotations, args.images, args.out, args.mode)


if __name__ == "__main__":
    main()
