"""
탐지 confidence에 따른 알림 판단 로직 (PRD 6장/9장 FR-4):
  - confidence >= 0.25: 로그/미니맵에 표시
  - confidence >= 0.50: 부저 알림까지 발생 (로그/미니맵 표시 포함)

여기서는 순수 판단 로직만 다룬다 (하드웨어 의존 없음, GPIO/LCD 없이 단위 테스트 가능).
실제 부저(GPIO)/LCD 출력은 Jetson 연결 후 별도 하드웨어 모듈(edge/notify/hardware.py 등,
미구현)에서 이 모듈이 반환하는 NotifyDecision을 받아 처리하도록 분리한다.
"""
from dataclasses import dataclass
from enum import Enum

LOG_THRESHOLD = 0.25
BUZZER_THRESHOLD = 0.5


class NotifyLevel(Enum):
    NONE = "none"
    LOG_ONLY = "log_only"  # log/minimap 표시만 (0.25 <= conf < 0.5)
    BUZZER = "buzzer"  # log/minimap + 부저 (conf >= 0.5)


@dataclass
class NotifyDecision:
    direction: str
    confidence: float
    level: NotifyLevel

    @property
    def should_log(self):
        return self.level in (NotifyLevel.LOG_ONLY, NotifyLevel.BUZZER)

    @property
    def should_buzz(self):
        return self.level == NotifyLevel.BUZZER


def decide(direction: str, confidence: float) -> NotifyDecision:
    if confidence >= BUZZER_THRESHOLD:
        level = NotifyLevel.BUZZER
    elif confidence >= LOG_THRESHOLD:
        level = NotifyLevel.LOG_ONLY
    else:
        level = NotifyLevel.NONE
    return NotifyDecision(direction=direction, confidence=confidence, level=level)


def decide_for_frame_results(frame_results: dict) -> list:
    """edge/camera/inference_pipeline.py의 FrameResult 딕셔너리({direction: FrameResult})를
    받아 채널별 판단 목록을 반환. FrameResult를 직접 import하지 않고 duck-typing으로 다뤄
    (frame_ok, max_confidence 속성만 있으면 됨) 카메라 모듈과 결합도를 낮춘다."""
    decisions = []
    for direction, result in frame_results.items():
        if not getattr(result, "frame_ok", False):
            continue
        conf = getattr(result, "max_confidence", 0.0)
        if conf <= 0:
            continue
        decisions.append(decide(direction, conf))
    return decisions


def main():
    samples = [0.0, 0.1, 0.25, 0.3, 0.49, 0.5, 0.7, 0.99]
    print("confidence -> level (should_log / should_buzz)")
    for conf in samples:
        d = decide("front", conf)
        print(f"  {conf:.2f} -> {d.level.value:<9} (log={d.should_log}, buzz={d.should_buzz})")


if __name__ == "__main__":
    main()
