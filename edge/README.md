# edge/

Jetson Orin Nano에서 동작하는 엣지 소프트웨어.

- `camera/` — 4채널 카메라 입력 및 YOLO11(TensorRT) 추론
- `gps/` — GPS 모듈(UART) 파싱 및 위치 스트리밍 (pyserial + pynmea2)
- `notify/` — 부저(GPIO) 및 LCD(PyQt) 알림
- `comm/` — 서버 전송(REST, hwpx 인수인계 문서 스펙 기준 `/api/ingest`) 및 오프라인
  큐잉(SQLite, `queue_store.py`). 부저는 로컬 confidence 판단(`notify/decision.py`)이
  최종 권한을 갖고, 서버 `/api/device/{id}/command`의 riskLevel은 참고용으로만 사용
  (`client.py`의 `reconcile_buzzer` 참고)
- `config/` — 임계값, 카메라 매핑 등 설정 파일. 서버 연동 장비 키 등 비밀값은
  `config/settings.py`가 프로젝트 루트의 `.env`(git 추적 안 됨, `config/.env.example` 참고)
  에서만 읽어오며, 코드에는 절대 하드코딩하지 않는다.

Phase 4~5에서 구현 예정 (PRD 12장 마일스톤 참고).
