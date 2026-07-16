# STACK.md — 기술 스택 및 개발 환경 확정 내역

Phase 1(프로젝트 세팅) 산출물. PRD 6장(이전 맥북 환경에서 확정된 결정)과 이번 RTX 3090 PC 환경에서 실제 검증한 버전을 함께 정리한다.

---

## 1. PRD 6장 확정 사항 (유지)

| 항목 | 결정 | 비고 |
|---|---|---|
| 탐지 모델 | YOLO11 → ONNX → TensorRT(FP16) | Jetson Orin Nano 공식 검증 조합. **최종 모델 크기: YOLO11n(n_full) 확정** — 아래 "1.1 최종 모델 확정" 참고 |
| 4채널 카메라 처리 | 독립 추론(스티칭 없음) | |
| 데이터 전략 | 공개데이터(occlusion) + 합성 occlusion aug(주력) + 자체촬영(검증) | |
| GPS | UART + pyserial/pynmea2, 2초 주기 | |
| 부저/LCD | GPIO 부저(Jetson 전용), LCD는 PyQt 경량 네이티브 UI | |
| 서버 | FastAPI + PostgreSQL/PostGIS | 팀원 A 담당, 이 문서 범위 밖 |
| 대시보드 | React + TypeScript + Kakao Map | |
| 통신 | Wi-Fi + HTTP REST, 서버→대시보드 WebSocket, 로컬 SQLite 오프라인 큐 | |
| 신뢰도 임계값(잠정) | 로그/미니맵 ≥0.25, 부저 ≥0.5 | 실측 후 재조정 예정 |
| 라벨링 방식 | 전체범위(full-extent) bbox | 스모크테스트로 확인됨 (PRD 8장) |

---

## 1.1 최종 모델 확정 (Phase 3 학습 완료 후)

- **YOLO11n(n_full)으로 최종 확정. YOLO11s(s_full) 학습은 진행하지 않음.**
- 근거:
  - n_full의 test set 성능이 이미 목표를 충분히 달성 (mAP50 0.803, occlusion 70% 구간에서
    베이스라인 대비 mAP50 +0.496 — MODEL_REPORT.md 3장 참고).
  - Jetson Orin Nano 엣지 실시간 추론에는 n이 s보다 유리 (파라미터 수↓, 추론 지연시간↓).
  - s_full 학습에 드는 추가 시간 대비, 이미 검증된 n_full 대비 기대 성능 향상폭이 크지
    않을 것으로 판단 (시간 대비 효율).
- 상세 판단 근거와 실험 데이터는 `MODEL_REPORT.md` 5장 참고.

---

## 2. GPU 환경 — PRD 가정과의 차이 (중요)

- **PRD 가정**: RTX 5080
- **실제 확인**: 이 PC에 장착된 GPU는 **RTX 5080이 아니라 RTX 3090** (Ampere, sm_86, VRAM 24GB)
  - `nvidia-smi` 확인: Driver 591.86, Driver-supported CUDA 13.1
- **결정**: 사용자 확인 하에 **RTX 3090 기준으로 진행**. RTX 5080(Blackwell, sm_120)은 별도 아키텍처로 최신 torch/CUDA 빌드가 필요해 호환 리스크가 더 크므로, 오히려 RTX 3090(Ampere)이 안정적인 조합으로 학습을 진행하기에 더 낫다.
- RTX 5080으로 실제 교체되는 시점이 오면, 이 문서의 "3. RTX 3090 환경 확정 버전"을 재검증 필요 (Blackwell은 CUDA 12.8+/최신 torch 빌드 요구 가능성).

---

## 3. RTX 3090 환경 확정 버전 (이번 단계에서 실측/검증 완료)

### 하드웨어/드라이버
| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 (Ampere, sm_86, 24GB) |
| 드라이버 | 591.86 (CUDA 13.1 지원) |

### Python
| 항목 | 값 | 비고 |
|---|---|---|
| Python | **3.12.10** (python.org 표준 배포판) | 프로젝트 전용 venv(`.venv/`) |

**중요 — Python 배포판 이슈**: 처음에는 시스템에 이미 설치돼 있던 Anaconda(`C:\ProgramData\anaconda3`, Python 3.11)를 venv 베이스로 사용했으나, `import torch` 시 `OSError: [WinError 1114] DLL initialization routine failed (c10.dll)`가 발생했다.
- 원인: Anaconda 배포판이 자체 번들한 **구버전 msvcp140.dll(2020, v14.27)**이 PATH상에서 발견되어, torch가 요구하는 최신 MSVC 런타임(v14.44, 시스템에 정상 설치돼 있었음)과 충돌 → 프로세스 내 액세스 위반(`0xc0000005`) 발생 (Windows 이벤트 로그로 확인).
- 조치: python.org 표준 설치본(Python 3.12.10, 사용자 계정 범위 설치, `C:\Users\user\AppData\Local\Programs\Python\Python312`)으로 venv를 재생성 → 문제 해결, `torch.cuda.is_available()` 정상 동작 확인.
- 참고: 기존 시스템 Python 3.9.13(python.org 배포판, Anaconda 아님)은 이 DLL 충돌은 없었으나 **이미 EOL(2025-10)**이고, 최신 torch(2.9.0+)가 cp39 wheel을 더 이상 제공하지 않아(2.8.0이 마지막) 배제.
- **팀 공유 시 주의**: 다른 팀원 PC에 Anaconda가 설치돼 있고 PATH에 잡혀 있다면 동일 문제가 재현될 수 있음. venv는 반드시 python.org 표준 배포판(Anaconda 아님) 기반으로 생성할 것.

### 딥러닝 스택
| 패키지 | 버전 | 비고 |
|---|---|---|
| torch | 2.13.0+cu130 | `--index-url https://download.pytorch.org/whl/cu130` |
| torchvision | 0.28.0+cu130 | 동일 인덱스 |
| ultralytics | 8.4.95 | YOLO11 포함, torch 버전 제약 없음(`torch>=1.8.0`) |

- CUDA 13.0 빌드(cu130) 선택 이유: 드라이버가 지원하는 CUDA 13.1과 가장 근접한 최신 빌드이며, torch 2.13.0(최신 안정판)이 cu130으로 배포됨. cu126 빌드도 RTX 3090(sm_86)에는 문제없이 동작하나, 향후 RTX 5080(Blackwell) 전환 시 더 유리한 최신 빌드를 기준으로 확정.
- 검증: `torch.cuda.is_available() == True`, GPU 행렬곱 연산 정상, `torch.cuda.get_device_name(0) == 'NVIDIA GeForce RTX 3090'`, `get_device_capability(0) == (8, 6)`.
- YOLO11n 사전학습 가중치(`yolo11n.pt`)로 GPU 추론 스모크테스트 완료 (`model/experiments`에서 실행, bus.jpg 샘플 이미지에서 사람 4명 + 버스 1대 정상 탐지, GPU inference ~49ms/image).

### 엣지 SW 관련 (PC에서 개발/테스트 가능한 부분)
| 패키지 | 버전 | 비고 |
|---|---|---|
| pyserial | 3.5 | GPS UART 파싱 |
| pynmea2 | 1.19.0 | NMEA 문장 파싱 |
| PyQt6 | 6.11.0 | LCD 알림 UI (PyQt5 대신 PyQt6로 확정 — 신규 프로젝트 기준 최신 유지보수 라인) |

### Jetson 전용 (Windows PC에는 설치 불가)
- `Jetson.GPIO` — 부저 GPIO 제어. Jetson Orin Nano 실기 확보 후 별도 설치.
- ONNX → TensorRT(FP16) 변환/추론 스택 — Jetson 실기 및 JetPack 버전 확정 후 별도 문서화 (PRD 13장 미확정 사항).

---

## 4. requirements.txt 사용법

```
py -3.12 -m venv .venv        # 반드시 python.org 표준 배포판 기준 (Anaconda 아님)
.venv\Scripts\pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt
```

`requirements.txt` 상단에 `--index-url`/`--extra-index-url`이 지정되어 있어 위 명령 한 줄로 torch(cu130)와 나머지 PyPI 패키지가 모두 설치된다.

---

## 5. 남은 미확정 사항 (PRD 13장, Phase 1에서 해소되지 않은 부분)

- Jetson Orin Nano 실물 확보 시점 및 JetPack/TensorRT 버전 → 확보 후 별도 검증
- CrowdHuman/WiderPerson 라이선스 서약 및 다운로드 절차 → Phase 2에서 진행
- 자체 촬영 일정(팀원 C, D 협의) → Phase 2에서 진행
- RTX 5080으로 실제 전환 시 이 문서 3장 재검증 필요
