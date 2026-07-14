# model/

YOLO11 기반 잔해 매몰 인체 탐지 모델 학습. 데이터셋 구축 계획 전체는
[docs/DATASET.md](../docs/DATASET.md) 참고.

- `data/raw/{coco,crowdhuman,widerperson,ochuman}/` — 공개 데이터셋 원본 (gitignore 처리)
- `data/processed/{coco,crowdhuman,widerperson,ochuman}/` — YOLO 포맷 변환 결과 (full-extent bbox, gitignore 처리)
- `data/synthetic/` — 합성 occlusion augmentation 결과물 + metadata.csv (gitignore 처리)
- `data/own_capture/` — 자체 촬영 원본(`raw/`) 및 라벨링 완료본(`labeled/`) (gitignore 처리, [SHOOTING_GUIDE.md](../docs/SHOOTING_GUIDE.md) 참고)
- `data/splits/` — train/val/test 목록 (재생성 가능, gitignore 처리)
- `data/dataset.yaml` — YOLO 학습 데이터 설정 (git 추적)
- `scripts/convert_to_yolo.py` — 소스별 원본 포맷 → YOLO 라벨 변환
- `scripts/occlusion_augment.py` — 합성 occlusion augmentation (30/50/70%)
- `scripts/split_dataset.py` — train/val/test 분할
- `experiments/` — 실험 결과, 스모크테스트 로그 (PRD 8장 이전 실험 참고)
- `weights/` — 학습된 가중치 (.gitignore 처리, 원격 저장소로 별도 관리)

데이터셋과 가중치는 이전 환경 복구 불가로 Phase 2~3에서 재구축.
