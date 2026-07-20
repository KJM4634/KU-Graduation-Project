"""
서버(/api/ingest)로 telemetry + frame 메시지를 전송하는 클라이언트. hwpx 인수인계
문서 스펙 기준: {incidentId, incidentName, incidentType, deviceId, deviceKey, messages:[...]}

부저 알림은 로컬 confidence 판단(edge/notify/decision.py, PRD FR-4)이 최종 권한을
가진다. 서버가 GET /api/device/{id}/command로 내려주는 riskLevel/buzzer 제안은
참고용으로만 받아온다 - PRD 10장 "네트워크 단절 시에도 로컬 탐지/알림 독립 동작"
요구사항 때문에, 서버 응답이 없어도(오프라인이어도) 로컬 알림은 그대로 동작해야 한다.

네트워크 단절 시에는 payload를 로컬 SQLite(queue_store.py)에 쌓아두고, 재연결되면
큐에 쌓인 것부터 순서대로(FIFO) 재전송한다 (PRD 6장).
"""
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edge.comm.queue_store import LocalStore
from edge.comm.schema import build_frame_message, build_telemetry_message
from edge.config.settings import load_server_config


class ServerClient:
    def __init__(self, config=None, store=None, timeout=3.0):
        self.config = config or load_server_config()
        self.store = store or LocalStore()
        self.timeout = timeout

    def _build_payload(self, messages):
        return {
            "incidentId": self.config.incident_id,
            "incidentName": self.config.incident_name,
            "incidentType": self.config.incident_type,
            "deviceId": self.config.device_id,
            "deviceKey": self.config.device_key,
            "messages": messages,
        }

    def _send_envelope(self, payload) -> bool:
        try:
            r = requests.post(
                f"{self.config.server_url}/api/ingest", json=payload, timeout=self.timeout
            )
            return r.ok
        except requests.RequestException:
            return False

    def flush_queue(self, max_items=None):
        """큐에 쌓인 이전 payload를 오래된 순서대로 재전송한다. 실패하면 그 지점에서
        멈춰(뒤엣것을 먼저 보내지 않음) 순서를 보존한다. 보낸 개수를 반환."""
        sent = 0
        while max_items is None or sent < max_items:
            item = self.store.peek_oldest()
            if item is None:
                break
            item_id, payload = item
            if self._send_envelope(payload):
                self.store.remove(item_id)
                sent += 1
            else:
                break
        return sent

    def send_tick(self, messages) -> str:
        """이번 tick의 메시지들을 전송한다. 큐에 이미 밀린 게 있으면(순서 보장을 위해)
        새 메시지를 바로 보내지 않고 큐 맨 뒤에 쌓는다 - 그래야 최신 위치가 오래된
        미전송 위치보다 서버에 먼저 도착해 순서가 뒤집히는 일이 없다.
        반환값: "sent" 또는 "queued"."""
        self.flush_queue()
        payload = self._build_payload(messages)
        if self.store.count() > 0:
            self.store.enqueue(payload)
            return "queued"
        if self._send_envelope(payload):
            return "sent"
        self.store.enqueue(payload)
        return "queued"

    def build_tick_messages(self, gps_fix, frame_results, frame_dims=None, heading_deg=0.0, telemetry_confidence=0.8):
        """GpsFix 1개 + {direction: FrameResult} 묶음으로 이번 tick의 telemetry+frame
        메시지 목록을 만든다. frame_dims: {direction: (width, height)}, 없으면 640x480."""
        now_ms = int(time.time() * 1000)
        messages = [
            build_telemetry_message(
                device_id=self.config.device_id,
                seq=self.store.next_seq("telemetry"),
                fix=gps_fix,
                confidence=telemetry_confidence,
                heading_deg=heading_deg,
                timestamp_ms=now_ms,
            )
        ]
        frame_dims = frame_dims or {}
        for direction, result in frame_results.items():
            if not getattr(result, "frame_ok", False):
                continue
            width, height = frame_dims.get(direction, (640, 480))
            messages.append(
                build_frame_message(
                    device_id=self.config.device_id,
                    seq=self.store.next_seq("frame"),
                    direction=direction,
                    frame_result=result,
                    width=width,
                    height=height,
                    timestamp_ms=now_ms,
                )
            )
        return messages

    def poll_command(self):
        """서버가 계산한 riskLevel/부저 제안을 조회. 실패하면 None을 반환하며,
        이는 로컬 알림 동작에 영향을 주지 않는다 (참고용)."""
        url = f"{self.config.server_url}/api/device/{self.config.device_id}/command"
        headers = {"X-Device-Key": self.config.device_key}
        try:
            r = requests.get(url, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            return None


def reconcile_buzzer(local_decisions, server_command):
    """부저를 울릴지는 로컬 confidence 판단(edge/notify/decision.py)이 최종 결정한다.
    서버의 riskLevel/message는 로그/참고용으로만 함께 반환하고, should_buzz를
    덮어쓰지 않는다 (네트워크 단절 시에도 로컬 알림이 독립 동작해야 하므로)."""
    should_buzz = any(getattr(d, "should_buzz", False) for d in local_decisions)
    return {
        "should_buzz": should_buzz,
        "local_reasons": [d.direction for d in local_decisions if getattr(d, "should_buzz", False)],
        "server_risk_level": (server_command or {}).get("riskLevel"),
        "server_message": (server_command or {}).get("message"),
    }
