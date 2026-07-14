# DSAR-Assist (Disaster Search And Rescue Assist)

재난 잔해지대 인명탐지 및 구조지원 시스템 — 졸업작품 프로젝트

## 문서
- 제품 요구사항: [docs/PRD.md](docs/PRD.md)
- 기술 스택 및 환경 확정 내역: [docs/STACK.md](docs/STACK.md)

## 폴더 구조
| 폴더 | 내용 |
|---|---|
| `edge/` | Jetson Orin Nano 엣지 소프트웨어 (카메라 추론, GPS, 부저/LCD 알림, 서버 통신) |
| `model/` | YOLO11 학습/데이터셋 구축 (공개데이터 + 합성 occlusion + 자체촬영) |
| `server/` | FastAPI + PostgreSQL/PostGIS 서버 (팀원 A 담당) |
| `dashboard/` | React + TypeScript + Kakao Map 관제센터 대시보드 |
| `docs/` | PRD, 스택 결정, 기타 문서 |

## 개발 환경
- 학습 PC: Windows, RTX 3090 (CUDA) — 상세는 `docs/STACK.md` 참고 (PRD상 RTX 5080 가정과 실제 장착 GPU가 달라 RTX 3090 기준으로 확정)
- 배포 타겟: Jetson Orin Nano (ONNX → TensorRT FP16)

## 개발 환경 세팅
```
py -3.12 -m venv .venv        # 반드시 python.org 표준 배포판 기준 (Anaconda venv 사용 시 DLL 충돌 발생, docs/STACK.md 참고)
.venv\Scripts\pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt
```

## 팀 구성
| 역할 | 담당 |
|---|---|
| 서버 개발 | 팀원 A |
| AI 모델링 + 엣지 SW + 대시보드 | 팀원 B (본인) |
| 하드웨어 조립 / 데이터 촬영 협조 | 팀원 C, D |
