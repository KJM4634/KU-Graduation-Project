"""
YOLO11 본 학습 스크립트 (Phase 3). DATASET.md 최종 분할(전체범위 full-extent bbox 라벨)로
COCO 사전학습 가중치를 파인튜닝한다.

사용 예:
    python train_yolo.py --model n --epochs 100 --patience 30 --imgsz 640
    python train_yolo.py --model s --epochs 100 --patience 30 --imgsz 640
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["n", "s"], required=True, help="YOLO11 모델 크기")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "model" / "data" / "dataset.yaml"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", default=-1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=0)
    args = parser.parse_args()

    weights = PROJECT_ROOT / "model" / "weights" / f"yolo11{args.model}.pt"
    model = YOLO(str(weights))

    model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(PROJECT_ROOT / "model" / "experiments" / "train"),
        name=f"{args.model}_full",
        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
