"""
YOLO11 .pt 가중치를 ONNX로 변환하고, onnxruntime으로 실제 추론이 되는지 검증한다.
TensorRT 변환은 Jetson 본체에서 진행 (PRD 6장: YOLO11 -> ONNX -> TensorRT(FP16)).

사용 예:
    python export_onnx.py --weights model/experiments/train/n_full/weights/best.pt \
        --test-image model/experiments/report_assets/../../../data/... (임의 person 이미지)
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO


def export(weights, imgsz=640, opset=12):
    model = YOLO(weights)
    onnx_path = model.export(format="onnx", imgsz=imgsz, simplify=True, opset=opset)
    return Path(onnx_path)


def letterbox(img, imgsz=640, color=(114, 114, 114)):
    """YOLO 표준 letterbox: 종횡비 유지한 채 리사이즈 + 패딩 (naive resize는 종횡비를
    왜곡해 .pt 대비 confidence가 낮게 나오는 문제가 있어 이 방식으로 맞춤)."""
    h, w = img.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), color, dtype=np.uint8)
    top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def preprocess(image_path, imgsz=640):
    img = cv2.imread(str(image_path))
    img_letterboxed = letterbox(img, imgsz)
    img_rgb = cv2.cvtColor(img_letterboxed, cv2.COLOR_BGR2RGB)
    tensor = img_rgb.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[None, ...]  # NCHW
    return np.ascontiguousarray(tensor), img.shape[:2]


def run_onnx_inference(onnx_path, image_path, imgsz=640, conf=0.25, iou=0.45):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    active_provider = session.get_providers()[0]

    tensor, _ = preprocess(image_path, imgsz)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tensor})
    pred = outputs[0]  # (1, 5, 8400): [xc, yc, w, h, conf] (단일 클래스라 클래스 점수 없이 5채널)

    pred = pred[0].transpose(1, 0)  # (8400, 5)
    scores = pred[:, 4]
    max_score = float(scores.max())

    # NMS 전 raw anchor 수 (참고용) vs NMS 후 최종 박스 수 (.pt와 비교 가능한 수치)
    n_raw = int((scores >= conf).sum())
    keep_mask = scores >= conf
    boxes_xywh = pred[keep_mask, :4]
    scores_kept = scores[keep_mask]
    boxes_xyxy = np.stack([
        boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2,
        boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2,
        boxes_xywh[:, 2], boxes_xywh[:, 3],  # cv2.dnn.NMSBoxes는 [x,y,w,h] 형식 기대
    ], axis=1)
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(), scores_kept.tolist(), score_threshold=conf, nms_threshold=iou
    )
    n_after_nms = len(indices)

    return {
        "provider": active_provider,
        "n_raw_over_conf": n_raw,
        "n_detections_after_nms": n_after_nms,
        "max_score": max_score,
        "output_shape": list(outputs[0].shape),
    }


def run_pt_inference(weights, image_path, conf=0.25):
    model = YOLO(weights)
    results = model.predict(source=str(image_path), conf=conf, verbose=False)
    return {"n_detections": len(results[0].boxes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--test-image", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    print(f"[1/3] {args.weights} -> ONNX 변환 중...")
    onnx_path = export(args.weights, args.imgsz)
    print(f"완료: {onnx_path}")

    print(f"\n[2/3] PyTorch(.pt) 추론: {args.test_image}")
    pt_result = run_pt_inference(args.weights, args.test_image, args.conf)
    print(pt_result)

    print(f"\n[3/3] ONNX(onnxruntime) 추론: {args.test_image}")
    onnx_result = run_onnx_inference(onnx_path, args.test_image, args.imgsz, args.conf)
    print(onnx_result)

    print(f"\n=== 비교 ===")
    print(f".pt 탐지 수: {pt_result['n_detections']} / "
          f"ONNX 탐지 수(NMS 후, conf>={args.conf}): {onnx_result['n_detections_after_nms']} "
          f"(NMS 전 raw anchor 수: {onnx_result['n_raw_over_conf']})")
    print(f"ONNX 최고 confidence: {onnx_result['max_score']:.3f}")
    print(f"ONNX 실행 provider: {onnx_result['provider']}")


if __name__ == "__main__":
    main()
