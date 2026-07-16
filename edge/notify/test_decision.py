"""decision.py 단위 테스트 (pytest 등 외부 의존성 없이 assert로 직접 실행).

    python test_decision.py
"""
from dataclasses import dataclass, field

from decision import NotifyLevel, decide, decide_for_frame_results


def test_thresholds():
    assert decide("front", 0.0).level == NotifyLevel.NONE
    assert decide("front", 0.24).level == NotifyLevel.NONE
    assert decide("front", 0.25).level == NotifyLevel.LOG_ONLY  # 경계값 포함
    assert decide("front", 0.49).level == NotifyLevel.LOG_ONLY
    assert decide("front", 0.5).level == NotifyLevel.BUZZER  # 경계값 포함
    assert decide("front", 0.99).level == NotifyLevel.BUZZER
    print("test_thresholds passed")


def test_should_log_should_buzz_flags():
    none = decide("front", 0.1)
    log_only = decide("front", 0.3)
    buzzer = decide("front", 0.7)

    assert not none.should_log and not none.should_buzz
    assert log_only.should_log and not log_only.should_buzz
    assert buzzer.should_log and buzzer.should_buzz
    print("test_should_log_should_buzz_flags passed")


@dataclass
class _FakeFrameResult:
    """edge/camera/inference_pipeline.py의 FrameResult를 흉내내는 최소 스텁
    (ultralytics import 없이 decide_for_frame_results만 테스트하기 위함)."""
    frame_ok: bool
    max_confidence: float = 0.0


def test_decide_for_frame_results():
    frame_results = {
        "front": _FakeFrameResult(frame_ok=True, max_confidence=0.8),   # buzzer
        "back": _FakeFrameResult(frame_ok=True, max_confidence=0.3),    # log_only
        "left": _FakeFrameResult(frame_ok=True, max_confidence=0.0),    # 탐지 없음 -> 제외
        "right": _FakeFrameResult(frame_ok=False),                      # 프레임 없음 -> 제외
    }
    decisions = decide_for_frame_results(frame_results)
    by_direction = {d.direction: d for d in decisions}

    assert len(decisions) == 2
    assert by_direction["front"].level == NotifyLevel.BUZZER
    assert by_direction["back"].level == NotifyLevel.LOG_ONLY
    assert "left" not in by_direction
    assert "right" not in by_direction
    print("test_decide_for_frame_results passed")


if __name__ == "__main__":
    test_thresholds()
    test_should_log_should_buzz_flags()
    test_decide_for_frame_results()
    print("\n모든 테스트 통과")
