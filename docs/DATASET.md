# DATASET.md — 데이터셋 구축 계획

Phase 2 산출물. PRD 6장(라벨링 방식)/7장(데이터셋 전략)/8장(초기 실험) 기준으로 실제 구축 절차와
스크립트 사용법을 정리한다.

---

## 1. 전체 전략 (PRD 7장 요약)

**공개 데이터셋 + 합성 occlusion augmentation(주력) + 자체 촬영(검증)** 3축 조합.
라벨링은 예외 없이 **전체범위(full-extent) bbox** 방식으로 통일한다 — 가려짐과 무관하게
사람의 실제 전체 범위를 라벨로 표시 (초기 스모크테스트에서 visible-only 대비 recall/mAP50
모두 확연히 우세함을 확인, PRD 6장/8장).

모든 소스는 단일 클래스 **person(class 0)** 으로 통일한다. 프로젝트 목표가 "잔해에 부분
가려진 사람 탐지"이지 사람의 유형 분류가 아니므로, 클래스는 늘리지 않는다.

---

## 2. 공개 데이터셋

### 2.1 소스별 요약

| 데이터셋 | 우선순위 | 다운로드 방식 | 자동화 | 라이선스 | 확보 현황 |
|---|---|---|---|---|---|
| COCO (person) | 상 (기반) | 공식 서버 직접 다운로드 | ✅ 스크립트로 자동화 가능 | 어노테이션 CC BY 4.0, 이미지는 Flickr 원저작자 권리(비상업 사용 문제없음) | ✅ **완료** (66,808장, 2026-07-14) |
| CrowdHuman | 상 (occlusion 학습 핵심) | Google Drive/Baidu Drive | ⚠️ 브라우저 수동 다운로드 필요 | 비상업 연구·교육 목적 한정, **이미지 재배포 금지** | ⬜ 미확보 |
| WiderPerson | 상 (부분가림 개념 최유사) | Google Drive/Baidu Drive | ⚠️ 브라우저 수동 다운로드 필요 | 비영리 학술 목적(non-commercial scientific use) 한정 | ⬜ 미확보 |
| OCHuman | 중~상 (극한 occlusion 검증) | Tsinghua 신청 폼 제출 | ⚠️ 폼 제출 필요, GitHub는 API 코드만 제공 | MIT (API 코드), 데이터 자체는 신청 시 조건 확인 필요 | ⬜ 미확보 |

**COCO 확보 내역**: instances_train2017.json + instances_val2017.json에서 person 카테고리가
포함된 이미지 66,808장(val2017 2,693 + train2017 64,115)을 개별 URL로 다운로드
(`model/scripts/download_coco_person_subset.py`), `convert_to_yolo.py coco`로 YOLO 라벨
변환까지 완료. `model/data/processed/coco/`에 이미지 66,808장 + 라벨 66,808개 (빈 라벨 0개).

> **재배포 금지 주의**: CrowdHuman/WiderPerson은 원본 이미지를 그대로 팀 저장소나 발표 자료에
> 포함해 공개하면 안 된다. 로컬 학습용으로만 사용하고, `.gitignore`로 `model/data/raw/`,
> `model/data/processed/`를 커밋 대상에서 제외했다 (아래 6장 참고).

### 2.2 다운로드 절차

**COCO** (자동화 완료, ✅ 확보됨):
```bash
# 1. 어노테이션 다운로드 및 압축 해제 (instances_train2017.json, instances_val2017.json을
#    model/data/raw/coco/annotations/ 에 배치)
curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# 2. person 카테고리가 포함된 이미지만 개별 URL로 다운로드 (전체 train2017.zip(18GB)을
#    통째로 받을 필요 없음 — http://images.cocodataset.org/{split}/{file_name} 패턴)
python model/scripts/download_coco_person_subset.py --out model/data/raw/coco/images --workers 32
```
- 병렬 다운로드 워커 수는 32가 최적점으로 실측됨 (64는 서버 쪽에서 오히려 처리량이
  21장/초 → 4장/초로 저하 — 과도한 동시 연결이 역효과를 낸 것으로 추정)
- 개별 이미지 다운로드에는 20초 타임아웃을 명시적으로 설정 (미설정 시 일부 죽은 연결에서
  무한 대기하는 사례 발생 확인)
- 실제로 val2017 2,693장 + train2017 64,115장 = **66,808장 전량 다운로드 완료**
  (2026-07-14, HTTP 503 등 일시적 오류로 실패한 7장은 재실행 시 자동 재시도되어 회수)
- 이후 `model/data/processed/coco/`로 YOLO 변환 완료 (이미지 66,808장, 라벨 66,808개, 빈 라벨 0개)

**CrowdHuman** (수동):
1. https://www.crowdhuman.org/download.html 방문
2. `CrowdHuman_train01.zip` ~ `train03.zip`, `CrowdHuman_val.zip`, `annotation_train.odgt`,
   `annotation_val.odgt` 다운로드 (Google Drive 또는 Baidu Drive 링크)
3. 압축 해제 후 이미지를 `model/data/raw/crowdhuman/images/`, odgt 파일을
   `model/data/raw/crowdhuman/`에 배치

**WiderPerson** (수동):
1. http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/ 방문 (접속이 원활하지 않을 수 있음 —
   재시도하거나 논문에 안내된 Google Drive/Baidu Drive(코드 `uq3u`) 링크 이용)
2. 이미지와 `{이미지명}.jpg.txt` 어노테이션을 각각
   `model/data/raw/widerperson/images/`, `model/data/raw/widerperson/annotations/`에 배치

**OCHuman** (수동, 신청 필요):
1. https://github.com/liruilong940607/OCHumanApi 에서 API 코드 확인
2. Tsinghua 신청 폼(`cg.cs.tsinghua.edu.cn/dataset/form.html?dataset=ochuman`) 제출 후 데이터 수령
3. `model/data/raw/ochuman/images/`, annotation json을 `model/data/raw/ochuman/`에 배치
4. **주의**: OCHuman bbox가 amodal(full-extent)인지 원 논문(Pose2Seg, CVPR 2019) 기준으로
   명확히 확인되지 않았다. 확인 전까지는 학습(train)에는 포함하지 않고
   **val/test 검증 용도로만 사용**한다 (`model/scripts/split_dataset.py`에 이미 0/50/50
   비율로 기본 설정됨).

### 2.3 YOLO 포맷 변환

```bash
# COCO
python model/scripts/convert_to_yolo.py coco \
  --annotations model/data/raw/coco/annotations/instances_train2017.json \
  --images model/data/raw/coco/images \
  --out model/data/processed/coco

# CrowdHuman
python model/scripts/convert_to_yolo.py crowdhuman \
  --odgt model/data/raw/crowdhuman/annotation_train.odgt \
  --images model/data/raw/crowdhuman/images \
  --out model/data/processed/crowdhuman

# WiderPerson
python model/scripts/convert_to_yolo.py widerperson \
  --annotations model/data/raw/widerperson/annotations \
  --images model/data/raw/widerperson/images \
  --out model/data/processed/widerperson

# OCHuman
python model/scripts/convert_to_yolo.py ochuman \
  --annotations model/data/raw/ochuman/annotations.json \
  --images model/data/raw/ochuman/images \
  --out model/data/processed/ochuman
```

변환 시 클래스 매핑 규칙:
- **COCO**: `category_id`가 `person`인 어노테이션만 사용
- **CrowdHuman**: `tag="person"`의 `fbox`(전체범위)만 사용, `tag="mask"`(무시 대상) 제외
- **WiderPerson**: 클래스 1(pedestrians)/2(riders)/3(partially-visible persons)만
  person으로 매핑. 4(ignore regions)/5(crowd)는 개별 full-extent 사람 라벨로 보기 어려워
  **제외** (YOLO가 "무시 영역"을 네이티브로 지원하지 않기 때문 — 알려진 한계로 기록)
- **OCHuman**: bbox를 그대로 사용하되 검증 전용으로 취급

각 변환 함수는 4종의 fixture(가짜 이미지+어노테이션)로 좌표 변환/클래스 필터링 로직을
검증 완료함 (실제 데이터셋 다운로드 전 단위 테스트로 확인).

---

## 3. 합성 Occlusion Augmentation

`model/scripts/occlusion_augment.py` — 절차적으로 생성한 잔해 텍스처(콘크리트/철근/먼지/천막)를
bbox 하단부에 합성해 occlusion 비율 30/50/70%를 제어한다 (PRD 7.2절).

```bash
python model/scripts/occlusion_augment.py \
  --images model/data/processed/coco/images \
  --labels model/data/processed/coco/labels \
  --out model/data/synthetic \
  --ratios 0.3 0.5 0.7 \
  --variants-per-image 3
```

- 텍스처는 의사(pseudo) Perlin 노이즈 기반으로 4종(concrete/rebar/dust/tarp) 절차적 생성,
  상단 경계를 들쭉날쭉하게 처리해 잔해 더미처럼 보이도록 함
- 목표 비율은 이분 탐색으로 실제 픽셀 기준 occlusion 비율에 **±5% 이내로 수렴**시킴
  (수렴 실패 시 가장 근접한 결과 사용, 최대 12회 시도)
- **라벨(full-extent bbox)은 원본 그대로 유지** — occlusion은 이미지에만 적용, 사람의 실제
  전체 범위 라벨은 변경하지 않음
- 모든 생성 기록은 `model/data/synthetic/metadata.csv`에 저장
  (`source_image, output_image, bbox_index, target_ratio, actual_ratio, texture_type`)
- 절차적 텍스처를 실제 촬영한 잔해 텍스처로 교체하면 합성 데이터 비율 상한을 60~70%까지
  완화 가능 (PRD 7.2절) — 향후 자체 촬영 진행 시 텍스처 이미지도 함께 확보해 반영 예정
- 테스트 이미지로 30/50/70% 각각 실제 occlusion 비율 0.31~0.72 범위로 수렴함을 확인 완료

학습셋 기준 합성 데이터 비율은 50~60%를 상한으로 권장 (질감 편향 위험, PRD 7.2절).

---

## 4. 자체 촬영 (검증 핵심 데이터)

- 실제 사람(마네킹 아님) 대상, 팀원 C·D가 진행
- 상세 절차/안전 원칙: **[docs/SHOOTING_GUIDE.md](SHOOTING_GUIDE.md)** (코딩 지식 불필요, 비개발
  팀원이 그대로 따라 할 수 있도록 작성)
- 목표 500~1,000장, occlusion 30/50/70% 균등 배분, 각도/거리/조명 다양성 확보
- 촬영 원본은 `model/data/own_capture/raw/`에 보관
- 라벨링(전체범위 bbox) 완료본은 `model/data/own_capture/labeled/{images,labels}/`에 배치
  (라벨링 도구는 팀 상황에 맞게 선정 — 예: LabelImg, CVAT, Roboflow 등 YOLO 포맷 export 가능한 도구)

---

## 5. Train/Val/Test 분할 (PRD 7.4절)

`model/scripts/split_dataset.py` — 소스별로 다른 비율을 적용해 `model/data/splits/`에
`train.txt`/`val.txt`/`test.txt`를 생성 (각 줄에 이미지 절대경로).

| 소스 | train | val | test |
|---|---|---|---|
| 공개+occlusion 서브셋 (coco/crowdhuman/widerperson) | 80% | 10% | 10% |
| OCHuman (검증 전용, 2장 참고) | 0% | 50% | 50% |
| 합성(synthetic) | 80% | 10% | 10% |
| 자체 촬영(own_capture) | 30% | 20% | **50%** (실사용 성능 증거이므로 test 비중 높게) |

```bash
python model/scripts/split_dataset.py --data-root model/data --seed 42
```

- 라벨(.txt)이 존재하는 이미지만 수집 (빈 라벨 = 사람 없는 배경 이미지도 유효한 negative
  샘플로 포함)
- 소스별로 셔플 후 비율에 맞춰 분할, 시드 고정으로 재현 가능
- 100장 규모 fixture로 실제 분할 비율이 정확히 일치함을 확인 완료 (예: coco 20장 →
  train16/val2/test2)

`model/data/dataset.yaml`이 위 분할 결과(`splits/*.txt`)를 참조하도록 이미 구성돼 있어,
데이터 배치 → 변환 → augmentation → 분할까지 마치면 바로 YOLO11 학습에 사용할 수 있다:
```bash
yolo detect train data=model/data/dataset.yaml model=yolo11n.pt epochs=100
```

---

## 6. 폴더 구조 및 버전관리 정책

```
model/data/
  raw/                    # 원본 다운로드 (gitignore 처리)
    coco/{images,annotations}/
    crowdhuman/
    widerperson/
    ochuman/
  processed/              # 소스별 YOLO 포맷 변환 결과 (gitignore 처리)
    {coco,crowdhuman,widerperson,ochuman}/{images,labels}/
  synthetic/              # occlusion augmentation 결과 + metadata.csv (gitignore 처리)
    images/ labels/ metadata.csv
  own_capture/            # 자체 촬영 (gitignore 처리)
    raw/                  # 촬영 원본
    labeled/{images,labels}/  # 라벨링 완료본
  splits/                 # train/val/test 목록 txt (gitignore 처리, 재생성 가능)
  dataset.yaml            # YOLO 학습 설정 (git 추적 — 가벼운 설정 파일)
```

`raw/`, `processed/`, `synthetic/`, `own_capture/`, `splits/`의 실제 데이터 파일은 용량이 크고
(CrowdHuman/WiderPerson은 재배포 금지 조건도 있음) 재현 가능하므로 git에 커밋하지 않는다
(`.gitignore`에 반영, 폴더 구조 유지를 위한 `.gitkeep`만 추적).

---

## 7. 남은 과제

- ~~COCO 개별 이미지 필터링 다운로드 스크립트 작성~~ → **완료** (66,808장 다운로드 및 YOLO 변환,
  2026-07-14)
- CrowdHuman/WiderPerson 실제 다운로드 및 배치 (팀 계정으로 Google Drive 접근 — 브라우저 수동
  다운로드 필요, 자동화 불가)
- OCHuman 신청 폼 제출 및 bbox 성격(amodal 여부) 원 논문 재확인
- 자체 촬영 일정 협의 (팀원 C, D) — SHOOTING_GUIDE.md 기준으로 진행
- 합성 텍스처를 실제 촬영 잔해 사진으로 교체하는 실험 (자체 촬영 진행 후)
- COCO 66,808장에 `occlusion_augment.py` 적용해 합성 데이터 생성 (다음 단계 후보)
- CrowdHuman/WiderPerson 확보 후 `convert_to_yolo.py` → `occlusion_augment.py` →
  `split_dataset.py` 전체 파이프라인을 수만 장 규모로 실행, PRD 8장 스모크테스트 결과 재검증
