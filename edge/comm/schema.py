"""
edge 파이프라인 내부 객체(GPS fix, 카메라 FrameResult/Detection)를 hwpx 인수인계
문서 기준 서버 전송 스키마(JSON)로 변환한다.

서버가 아직 확정하지 않은 부분은 필드 자체는 넣되 값은 None으로 채운다:
  - pose.x/y/z: 서버는 VO(Visual Odometry) 기반 로컬 좌표를 언급했으나, 팀원A 확인
    전까지는 GPS lat/lon을 그대로 태우는 임시 처리로 둔다 (gps_fix_to_pose 참고).
    확인 후 변환 방식이 바뀌면 이 함수만 교체하면 된다.
  - metrics.rangeM, detections[].rangeM: 거리 추정 로직 미구현.
  - detections[].inferenceMs: 현재 추론 파이프라인이 채널별 소요시간을 기록하지 않음.
"""

# 서버는 방향 이름으로 "rear"를 기대하지만, 카메라 모듈(edge/camera/capture.py)의
# DIRECTIONS는 "back"을 쓴다 - 전송 경계에서만 변환한다.
DIRECTION_TO_CAMERA_ID = {
    "front": "front",
    "back": "rear",
    "left": "left",
    "right": "right",
}


def xyxy_to_xywh(xyxy):
    """[x1, y1, x2, y2] -> [x, y, width, height]"""
    x1, y1, x2, y2 = xyxy
    return [x1, y1, x2 - x1, y2 - y1]


def gps_fix_to_pose(fix, heading_deg=0.0):
    """GPS lat/lon을 서버 pose 필드({x,y,z,headingDeg})에 임시로 그대로 태운다.
    서버가 VO 기반 로컬 좌표를 요구하는 것으로 확인되면 이 함수만 교체하면 된다."""
    return {
        "x": fix.lat,
        "y": fix.lon,
        "z": 0,
        "headingDeg": heading_deg,
    }


def detection_to_payload(detection, range_m=None):
    """카메라 파이프라인의 Detection(direction, confidence, xyxy)을 서버 detections[]
    항목으로 변환. 우리 모델은 사람 단일 클래스만 학습했으므로 kind/type은 고정값."""
    return {
        "kind": "victim",
        "type": "person_candidate",
        "confidence": round(float(detection.confidence), 4),
        "rangeM": range_m,
        "bbox": xyxy_to_xywh(detection.xyxy),
        "label": "person",
        "modelName": "yolo-person",
        "inferenceMs": getattr(detection, "inference_ms", None),
    }


def build_telemetry_message(device_id, seq, fix, confidence=0.8, heading_deg=0.0, timestamp_ms=None):
    return {
        "type": "telemetry",
        "deviceId": device_id,
        "seq": seq,
        "clientEventId": f"{device_id}-telemetry-{seq}",
        "timestamp": timestamp_ms,
        "pose": gps_fix_to_pose(fix, heading_deg),
        "confidence": confidence,
    }


def build_frame_message(device_id, seq, direction, frame_result, width, height, timestamp_ms=None):
    camera_id = DIRECTION_TO_CAMERA_ID.get(direction, direction)
    detections = [detection_to_payload(d) for d in frame_result.detections]
    return {
        "type": "frame",
        "deviceId": device_id,
        "seq": seq,
        "clientEventId": f"{device_id}-{camera_id}-{seq}",
        "cameraId": camera_id,
        "timestamp": timestamp_ms,
        "width": width,
        "height": height,
        "metrics": {"rangeM": None},
        "detections": detections,
    }
