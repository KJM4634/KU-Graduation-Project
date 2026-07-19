"""
카메라 입력 추상화. 실제 Jetson CSI 카메라 4대가 연결되기 전까지, 웹캠 1대 또는
저장된 샘플 영상 4개로 "4채널 독립 입력"을 시뮬레이션한다.

핵심 설계: `CameraSource`라는 공통 인터페이스(`read()`, `release()`)만 지키면 어떤
입력이든(웹캠/영상파일/실제 CSI 카메라) 동일한 방식으로 파이프라인에 꽂을 수 있다.
나중에 실제 카메라로 교체할 때는 `RealCsiCameraSource` 같은 클래스만 새로 만들어
`build_rig_from_real_cameras()`에 연결하면 되고, 추론 파이프라인(inference_pipeline.py)
쪽 코드는 전혀 손댈 필요가 없다.
"""
from abc import ABC, abstractmethod

import cv2

DIRECTIONS = ["front", "back", "left", "right"]  # 전, 후, 좌, 우


class CameraSource(ABC):
    """모든 카메라 입력이 구현해야 하는 최소 인터페이스."""

    @abstractmethod
    def read(self):
        """프레임 1장(np.ndarray, BGR) 또는 읽기 실패 시 None을 반환."""
        raise NotImplementedError

    def release(self):
        pass


class VideoFileCameraSource(CameraSource):
    """저장된 샘플 영상 파일을 카메라처럼 순환 재생하며 제공."""

    def __init__(self, path, loop=True):
        self.path = str(path)
        self.loop = loop
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"영상을 열 수 없음: {self.path}")

    def read(self):
        ret, frame = self.cap.read()
        if not ret and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()


class WebcamCameraSource(CameraSource):
    """실제 웹캠 1대. 4채널 전부에 이 프레임을 공급하면(구조 검증용) 실제 4방향
    화면은 아니지만, 파이프라인이 4개 입력을 "각각 독립적으로" 처리하는 구조 자체는
    동일하게 검증할 수 있다."""

    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"웹캠(index={index})을 열 수 없음")

    def read(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()


class FourChannelRig:
    """4방향(전/후/좌/우) 카메라를 한데 묶어 매 프레임마다 4장을 동시에 가져온다."""

    def __init__(self, sources: dict):
        missing = set(DIRECTIONS) - set(sources.keys())
        if missing:
            raise ValueError(f"누락된 방향: {missing}")
        self.sources = sources

    def read_all(self) -> dict:
        """{'front': frame, 'back': frame, 'left': frame, 'right': frame} 반환.
        읽기 실패한 채널은 값이 None."""
        return {direction: src.read() for direction, src in self.sources.items()}

    def release_all(self):
        seen = set()
        for src in self.sources.values():
            if id(src) not in seen:  # 웹캠 모드에서 동일 소스를 4번 공유하는 경우 중복 release 방지
                src.release()
                seen.add(id(src))


def build_rig_from_videos(video_paths: dict) -> FourChannelRig:
    """video_paths: {'front': 'path/a.mp4', 'back': ..., 'left': ..., 'right': ...}"""
    sources = {d: VideoFileCameraSource(p) for d, p in video_paths.items()}
    return FourChannelRig(sources)


def build_rig_from_webcam(index=0) -> FourChannelRig:
    """웹캠 1대로 4채널 구조를 시뮬레이션 (실제 4방향이 아니라 동일 프레임 공유)."""
    shared = WebcamCameraSource(index)
    sources = {d: shared for d in DIRECTIONS}
    return FourChannelRig(sources)


def build_rig_from_webcam_indices(indices: dict) -> FourChannelRig:
    """실제 카메라 4대를 방향별로 다른 index에 연결 (Jetson 실배포용).
    indices: {'front': 0, 'back': 1, 'left': 2, 'right': 3} 형태."""
    sources = {d: WebcamCameraSource(i) for d, i in indices.items()}
    return FourChannelRig(sources)
