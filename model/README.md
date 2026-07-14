# model/

YOLO11 기반 잔해 매몰 인체 탐지 모델 학습.

- `data/raw/` — 공개 데이터셋 원본 (COCO, CrowdHuman, WiderPerson, OCHuman 등)
- `data/processed/` — 라벨 변환/정제 후 학습용 데이터 (full-extent bbox)
- `data/synthetic/` — 합성 occlusion augmentation 결과물
- `scripts/` — 데이터 전처리, 합성 augmentation, 학습/평가 스크립트
- `experiments/` — 실험 결과, 스모크테스트 로그 (PRD 8장 이전 실험 참고)
- `weights/` — 학습된 가중치 (.gitignore 처리, 원격 저장소로 별도 관리)

데이터셋과 가중치는 이전 환경 복구 불가로 Phase 2~3에서 재구축.
