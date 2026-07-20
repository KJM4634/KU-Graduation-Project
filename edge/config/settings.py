"""
서버 연동에 필요한 장비 키 등 비밀값은 코드에 하드코딩하지 않고, 프로젝트 루트의
`.env` 파일(.gitignore로 커밋 제외됨) 또는 환경변수에서만 읽는다.

사용법:
    1. `edge/config/.env.example`을 프로젝트 루트에 `.env`로 복사
    2. 실제 SERVER_URL / DEVICE_ID / DEVICE_KEY 값으로 채움 (담당 팀원에게 전달받은 값)
    3. 코드에서는 `load_server_config()`만 호출
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"환경변수 {name}가 설정되지 않았습니다. 프로젝트 루트에 .env 파일을 만들고 "
            f"(edge/config/.env.example 참고) {name}를 채워주세요."
        )
    return value


class ServerConfig:
    """서버 연동에 필요한 값들. 생성 시점에 필수값 누락 여부를 바로 검증한다."""

    def __init__(self):
        self.server_url = _require("SERVER_URL").rstrip("/")
        self.device_id = _require("DEVICE_ID")
        self.device_key = _require("DEVICE_KEY")
        self.incident_id = os.environ.get("INCIDENT_ID", "INC-DEMO-001")
        self.incident_name = os.environ.get("INCIDENT_NAME", "Demo disaster field")
        self.incident_type = os.environ.get("INCIDENT_TYPE", "Fire/collapse search")


def load_server_config() -> ServerConfig:
    return ServerConfig()
