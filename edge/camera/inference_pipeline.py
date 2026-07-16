"""
4채널 독립 추론 파이프라인. PRD 6장: "독립 추론(스티칭 없음), 카메라=탐지방향 자연 매핑".

각 방향(전/후/좌/우) 프레임을 서로 완전히 독립적으로 YOLO에 통과시키고, 탐지 결과에
어느 채널(방향)에서 나왔는지를 그대로 태그해서 반환한다. 카메라 입력은
`capture.py`의 `CameraSource` 인터페이스로 분리돼 있어, 실제 카메라로 교체돼도
이 파일은 수정할 필요가 없다.
"""
import argparse
import time
from dataclasses import dataclass, field

from ultralytics import YOLO

from capture import DIRECTIONS, build_rig_from_videos, build_rig_from_webcam


@dataclass
class Detection:
    direction: str
    confidence: float
    xyxy: list


@dataclass
class FrameResult:
    direction: str
    frame_ok: bool
    detections: list = field(default_factory=list)

    @property
    def max_confidence(self):
        return max((d.confidence for d in self.detections), default=0.0)


class FourChannelInferencePipeline:
    def __init__(self, model_path, imgsz=640, conf=0.25, device=0):
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.device = device

    def process_channel(self, direction, frame):
        if frame is None:
            return FrameResult(direction=direction, frame_ok=False)
        result = self.model.predict(
            frame, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False
        )[0]
        detections = [
            Detection(direction=direction, confidence=float(box.conf[0]), xyxy=box.xyxy[0].tolist())
            for box in result.boxes
        ]
        return FrameResult(direction=direction, frame_ok=True, detections=detections)

    def process_all(self, frames: dict) -> dict:
        """frames: {'front': frame, ...} -> {'front': FrameResult, ...}
        4방향을 순차적으로(각 채널은 서로의 결과에 영향받지 않고 독립적으로) 처리."""
        return {direction: self.process_channel(direction, frame) for direction, frame in frames.items()}


def summarize(results: dict) -> str:
    lines = []
    for direction in DIRECTIONS:
        r = results.get(direction)
        if r is None or not r.frame_ok:
            lines.append(f"  {direction:>5}: 프레임 없음")
            continue
        if not r.detections:
            lines.append(f"  {direction:>5}: 탐지 없음")
        else:
            best = max(r.detections, key=lambda d: d.confidence)
            lines.append(f"  {direction:>5}: {len(r.detections)}건 탐지 (최고 conf={best.confidence:.2f})")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["videos", "webcam"], default="videos")
    parser.add_argument("--video-front", default=None)
    parser.add_argument("--video-back", default=None)
    parser.add_argument("--video-left", default=None)
    parser.add_argument("--video-right", default=None)
    parser.add_argument("--webcam-index", type=int, default=0)
    parser.add_argument("--model", default="model/weights/dsar_n_full.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iterations", type=int, default=10, help="테스트용 반복 횟수 (0=무한)")
    parser.add_argument("--interval", type=float, default=0.0, help="반복 간 대기(초)")
    args = parser.parse_args()

    if args.source == "videos":
        video_paths = {
            "front": args.video_front, "back": args.video_back,
            "left": args.video_left, "right": args.video_right,
        }
        missing = [d for d, p in video_paths.items() if not p]
        if missing:
            raise SystemExit(f"--source videos 사용 시 --video-{{{','.join(missing)}}} 를 지정해야 함")
        rig = build_rig_from_videos(video_paths)
    else:
        rig = build_rig_from_webcam(args.webcam_index)

    pipeline = FourChannelInferencePipeline(args.model, conf=args.conf)

    i = 0
    try:
        while args.iterations == 0 or i < args.iterations:
            frames = rig.read_all()
            results = pipeline.process_all(frames)
            print(f"--- frame {i} ---")
            print(summarize(results))
            i += 1
            if args.interval > 0:
                time.sleep(args.interval)
    finally:
        rig.release_all()


if __name__ == "__main__":
    main()
