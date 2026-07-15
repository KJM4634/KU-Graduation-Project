"""
주어진 가중치로 특정 이미지 서브셋(txt 목록)에 대해 YOLO val을 실행하고
precision/recall/mAP50/mAP50-95를 반환한다.

사용 예:
    python evaluate_model.py --weights model/experiments/train/n_full/weights/best.pt \
        --subset model/data/splits/eval_subsets/test_synthetic_occ50.txt \
        --imgsz 640
"""
import argparse
import json
import tempfile
from pathlib import Path

from ultralytics import YOLO


def make_subset_yaml(subset_txt, tmp_dir):
    subset_txt = Path(subset_txt).resolve()
    yaml_path = Path(tmp_dir) / f"{subset_txt.stem}_eval.yaml"
    yaml_path.write_text(
        f"train: {subset_txt}\nval: {subset_txt}\ntest: {subset_txt}\nnc: 1\nnames: [\"person\"]\n",
        encoding="utf-8",
    )
    return yaml_path


def evaluate(weights, subset_txt, imgsz=640, batch=32, device=0):
    n_images = len([l for l in Path(subset_txt).read_text(encoding="utf-8").splitlines() if l.strip()])
    if n_images == 0:
        return {"n_images": 0, "precision": None, "recall": None, "map50": None, "map50_95": None}

    with tempfile.TemporaryDirectory() as tmp_dir:
        yaml_path = make_subset_yaml(subset_txt, tmp_dir)
        model = YOLO(str(weights))
        metrics = model.val(data=str(yaml_path), split="val", imgsz=imgsz, batch=batch,
                             device=device, verbose=False, plots=False)

    return {
        "n_images": n_images,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default=0)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    result = evaluate(args.weights, args.subset, args.imgsz, args.batch, args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
