"""
client.py 통합 테스트. 실제 서버(hwpx 인수인계 문서의 데모 서버)에는 절대 요청을
보내지 않는다 - 대신 로컬에 임시 mock HTTP 서버를 띄워 아래 흐름을 검증한다.

    python test_client.py

검증 내용:
  1) 서버가 응답 안 할 때(500) -> send_tick()이 로컬 SQLite에 큐잉되는지
  2) 서버가 복구된 뒤 -> 큐에 쌓인 것부터 순서대로(FIFO) 재전송되는지
  3) GPS lat/lon -> pose.x/y 임시 매핑, bbox xyxy->xywh, back->rear 변환 확인
  4) 큐가 비어있을 때는 즉시 전송되는지
  5) 서버 riskLevel과 무관하게 로컬 confidence 판단이 부저 최종 권한을 갖는지
"""
import json
import tempfile
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

from client import ServerClient, reconcile_buzzer
from queue_store import LocalStore


# edge/camera, edge/notify 모듈을 직접 import하지 않고(ultralytics 등 무거운 의존성
# 회피), 동일한 인터페이스(duck-typing)만 흉내내는 최소 스텁.

@dataclass
class _FakeGpsFix:
    lat: float
    lon: float
    timestamp: str = "12:00:00"


@dataclass
class _FakeDetection:
    direction: str
    confidence: float
    xyxy: list


@dataclass
class _FakeFrameResult:
    frame_ok: bool
    detections: list = field(default_factory=list)


@dataclass
class _FakeNotifyDecision:
    direction: str
    should_buzz: bool


class _MockIngestHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        if self.server.mode == "down":
            self.send_response(500)
            self.end_headers()
            return
        self.server.received.append(body)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "deviceId": "H1",
            "riskLevel": 3,
            "buzzer": {"enabled": True, "pattern": "critical", "intervalMs": 300},
            "message": "위험 접근 주의",
        }).encode("utf-8"))


def start_mock_server():
    httpd = HTTPServer(("127.0.0.1", 0), _MockIngestHandler)
    httpd.mode = "down"
    httpd.received = []
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def test_offline_queue_then_flush_in_order():
    httpd = start_mock_server()
    port = httpd.server_address[1]

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_queue.sqlite3"
        config = SimpleNamespace(
            server_url=f"http://127.0.0.1:{port}",
            device_id="H1",
            device_key="test-key",
            incident_id="INC-TEST-001",
            incident_name="Test incident",
            incident_type="Test",
        )
        client = ServerClient(config=config, store=LocalStore(db_path))

        # 1) 서버 다운 상태에서 3틱 전송 시도 -> 전부 큐잉돼야 함
        httpd.mode = "down"
        fixes = [_FakeGpsFix(37.1 + i * 0.001, 127.1 + i * 0.001) for i in range(3)]
        for fix in fixes:
            frame_results = {
                "front": _FakeFrameResult(frame_ok=True, detections=[
                    _FakeDetection("front", 0.8, [10, 10, 50, 90]),
                ]),
                "back": _FakeFrameResult(frame_ok=False),
            }
            messages = client.build_tick_messages(fix, frame_results)
            status = client.send_tick(messages)
            assert status == "queued", f"expected queued, got {status}"

        assert client.store.count() == 3, f"expected 3 queued, got {client.store.count()}"
        assert len(httpd.received) == 0, "서버가 다운인데 뭔가 도착함"
        print("1) 오프라인 큐잉 확인: 3건 모두 큐에 쌓임, 서버 미수신")

        # 2) 서버 복구 -> flush_queue()로 순서대로 재전송돼야 함
        httpd.mode = "up"
        sent = client.flush_queue()
        assert sent == 3, f"expected 3 flushed, got {sent}"
        assert client.store.count() == 0, "큐가 비어야 하는데 남아있음"
        assert len(httpd.received) == 3, f"서버가 3건 받아야 하는데 {len(httpd.received)}건 수신"

        received_seqs = [env["messages"][0]["seq"] for env in httpd.received]
        assert received_seqs == [1, 2, 3], f"순서가 뒤바뀜: {received_seqs}"
        print("2) 재연결 후 재전송 확인: 3건 모두 순서대로(seq 1,2,3) 도착")

        first_pose = httpd.received[0]["messages"][0]["pose"]
        assert first_pose["x"] == fixes[0].lat and first_pose["y"] == fixes[0].lon
        print("3) GPS lat/lon -> pose.x/y 임시 매핑 확인")

        frame_msgs = [m for env in httpd.received for m in env["messages"] if m["type"] == "frame"]
        assert frame_msgs[0]["cameraId"] == "front"
        assert frame_msgs[0]["detections"][0]["bbox"] == [10, 10, 40, 80]
        assert frame_msgs[0]["detections"][0]["kind"] == "victim"
        assert frame_msgs[0]["detections"][0]["type"] == "person_candidate"
        print("4) bbox 변환(xyxy->xywh) 및 kind/type 고정값 확인")

        fix4 = _FakeGpsFix(37.2, 127.2)
        messages4 = client.build_tick_messages(fix4, {"front": _FakeFrameResult(frame_ok=False)})
        status4 = client.send_tick(messages4)
        assert status4 == "sent", f"expected sent, got {status4}"
        assert len(httpd.received) == 4
        print("5) 큐가 비어있을 때는 즉시 전송 확인")

        client.store.close()

    httpd.shutdown()


def test_reconcile_buzzer_local_priority():
    local_decisions = [
        _FakeNotifyDecision(direction="front", should_buzz=False),
        _FakeNotifyDecision(direction="back", should_buzz=False),
    ]
    server_command = {"riskLevel": 3, "message": "위험 접근 주의"}
    result = reconcile_buzzer(local_decisions, server_command)
    assert result["should_buzz"] is False
    assert result["server_risk_level"] == 3
    assert result["server_message"] == "위험 접근 주의"
    print("6) 서버 riskLevel=3이어도 로컬 판단(False)이 최종 결정임을 확인")

    local_decisions2 = [_FakeNotifyDecision(direction="right", should_buzz=True)]
    result2 = reconcile_buzzer(local_decisions2, None)
    assert result2["should_buzz"] is True
    assert result2["server_risk_level"] is None
    print("7) 서버 응답이 없어도(오프라인) 로컬 판단만으로 should_buzz=True 유지 확인")


if __name__ == "__main__":
    test_offline_queue_then_flush_in_order()
    test_reconcile_buzzer_local_priority()
    print("\n모든 테스트 통과")
