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
| CrowdHuman | 상 (occlusion 학습 핵심) | Google Drive/Baidu Drive (또는 Kaggle 미러) | ⚠️ 브라우저 수동 다운로드 필요 | 비상업 연구·교육 목적 한정, **이미지 재배포 금지** | ✅ **완료** (train 15,000 + val 4,370 = 19,370장, 2026-07-15, Kaggle 경로로 확보) |
| WiderPerson | 상 (부분가림 개념 최유사) | Google Drive/Baidu Drive | ⚠️ 브라우저 수동 다운로드 필요 | 비영리 학술 목적(non-commercial scientific use) 한정 | ✅ **완료** (train 8,000 + val 1,000 = 9,000장, 공식 사이트 경유, 2026-07-15) |
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

**CrowdHuman** (수동, ✅ 확보됨):
1. https://www.crowdhuman.org/download.html 방문 (또는 동일 원본을 재배포하는 Kaggle 미러 —
   이번엔 Kaggle 경로로 확보, 실제 규모 검증 결과 아래 참고)
2. `CrowdHuman_train01.zip` ~ `train03.zip`, `CrowdHuman_val.zip`, `annotation_train.odgt`,
   `annotation_val.odgt` 다운로드 (Google Drive 또는 Baidu Drive 링크)
3. 압축 해제 후 이미지를 `model/data/raw/crowdhuman/images/`, odgt 파일을
   `model/data/raw/crowdhuman/`에 배치 (이번 확보본은 `Images/`(train), `Images_val/`(val)로
   분리된 구조라 변환 시 두 번 실행함, 아래 참고)

**CrowdHuman 검증 결과** (2026-07-15):
- 이미지 파일 수: train `Images/` 15,000장, val `Images_val/` 4,370장
- odgt 레코드 수: `annotation_train.odgt` 15,000줄, `annotation_val.odgt` 4,370줄 — **완전히 일치**
- odgt ID ↔ 이미지 파일명 교차 검증: 양방향 누락 **0건** (odgt에만 있는 이미지 0개, 이미지에만
  있고 odgt에 없는 경우 0개)
- 총 gtboxes: train 438,792개(person 352,983 + mask/ignore 85,809), val 127,716개(person
  103,115 + mask/ignore 24,601) — person당 이미지 밀도(약 23.5명/장)가 공식 CrowdHuman 통계와
  부합해 정상 데이터로 판단

**WiderPerson** (수동, ✅ 확보됨):
1. http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/ 방문 (접속이 원활하지 않을 수 있음 —
   재시도하거나 논문에 안내된 Google Drive/Baidu Drive(코드 `uq3u`) 링크 이용). 이번엔 공식
   사이트 안내에 따라 Google Drive 경유로 정상 확보 (910MB)
2. 이미지와 `{이미지명}.jpg.txt` 어노테이션을 각각
   `model/data/raw/widerperson/Images/`, `model/data/raw/widerperson/Annotations/`에 배치
   (원본 zip 구조 그대로 `Images`/`Annotations`로 대문자 시작 — Windows 탐색기로 `Annotations`
   폴더만 뒤늦게 복사하다가 "다른 프로그램에서 파일 사용 중" 오류가 나서 PowerShell
   `robocopy`로 대신 복사해 해결)

**WiderPerson 검증 결과** (2026-07-15):
- `Images/` 13,382장 = train.txt(8,000) + val.txt(1,000) + test.txt(4,382) 합계와 정확히 일치
  (`wc -l`은 마지막 줄 개행 누락으로 각각 1씩 적게 표시되었을 뿐, 실제 줄 수는 정확함)
- `Annotations/` 9,000개 = train+val(8,000+1,000)과 정확히 일치, test 이미지는 원래
  라벨을 공개하지 않는 정책(원본 데이터셋 자체 정책)이라 정상
- ID 교차 검증: `Images` ↔ (train+val+test 목록), `Annotations` ↔ (train+val 목록) 모두
  **불일치 0건**, test 이미지 중 Annotations가 존재하는 경우도 0건
- Annotation 포맷: 각 파일 첫 줄이 박스 개수(N), 이후 `class_label x1 y1 x2 y2` — 공식
  포맷과 정확히 일치, 형식 오류 라인 0개
- 전체 273,475박스 중 class 분포: 1(pedestrians) 178,593 / 2(riders) 1,659 /
  3(partially-visible persons) 79,934 / 4(ignore regions) 3,649 / 5(crowd) 9,640 —
  부분가림 클래스(3)가 뚜렷하게 존재함을 확인

**OCHuman** (수동, 신청 필요):
1. https://github.com/liruilong940607/OCHumanApi 에서 API 코드 확인
2. Tsinghua 신청 폼(`https://cg.cs.tsinghua.edu.cn/dataset/form.html?dataset=ochuman`) 제출 후
   데이터 수령 — 2026-07-15 기준 GitHub README에서 이 URL을 재확인함. 자동 크롤링으로 접속 시
   "No dataset available for your request"가 떠서 링크가 깨진 것처럼 보였으나, 페이지가
   `?dataset=ochuman` 쿼리를 자바스크립트로 읽어 내용을 바꾸는 방식이라 자동 크롤러가 제대로
   렌더링하지 못한 것으로 추정 — **반드시 실제 브라우저로 직접 접속해서 확인할 것**
   - 필요 입력 정보(Tsinghua CG 그룹 동일 계열 폼 기준): 이름, 이메일, 소속
   - 제공 파일: 이미지(약 667MB) + COCO 스타일 어노테이션(val/test 분할 포함)
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

# CrowdHuman (train/val 이미지 폴더가 분리돼 있으면 두 번 실행 — 같은 --out으로)
python model/scripts/convert_to_yolo.py crowdhuman \
  --odgt model/data/raw/crowdhuman/annotation_train.odgt \
  --images model/data/raw/crowdhuman/Images \
  --out model/data/processed/crowdhuman
python model/scripts/convert_to_yolo.py crowdhuman \
  --odgt model/data/raw/crowdhuman/annotation_val.odgt \
  --images model/data/raw/crowdhuman/Images_val \
  --out model/data/processed/crowdhuman

# WiderPerson (원본 zip 구조 그대로 Images/Annotations 대문자 시작)
python model/scripts/convert_to_yolo.py widerperson \
  --annotations model/data/raw/widerperson/Annotations \
  --images model/data/raw/widerperson/Images \
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

**CrowdHuman 변환 결과** (2026-07-15): `model/data/processed/crowdhuman/`에 이미지
19,370장(train 15,000 + val 4,370) + 라벨 19,370개, 빈 라벨 0개. `tag="mask"` 및
`extra.ignore=1`인 박스는 제외하고 `fbox`(전체범위)만 라벨로 사용.

**WiderPerson 변환 결과** (2026-07-15): `model/data/processed/widerperson/`에 이미지
9,000장 + 라벨 9,000개, 빈 라벨 0개. 변환 후 person 박스 260,183개 (클래스 1/2/3 원본
합계 260,186과 3개 차이 — 이미지 경계로 클리핑 후 면적이 0이 된 퇴화 박스가 걸러진 것으로,
무시 가능한 수준). 클래스 4(ignore regions)/5(crowd) 3,649+9,640=13,289개는 제외됨.

**WiderPerson class 3(부분가림) 서브셋**: class 3 박스를 하나 이상 포함하는 이미지
**6,861장 (76.2%)** — `model/data/processed/widerperson/partial_visible_subset.txt`에
이미지 ID 목록 저장. WiderPerson은 "partially-visible person" 전용 클래스를 제공한다는
PRD 7.1절 설명대로, 부분가림 샘플이 상당한 비중을 차지함을 확인.

### 2.4 Occlusion 비율 서브셋 분석 (CrowdHuman)

`model/scripts/analyze_crowdhuman_occlusion.py`로 vbox(visible)/fbox(full) 면적비 기반
occlusion_ratio(= 1 - vbox_area/fbox_area)를 계산해 PRD 7장의 30/50/70% 구간에 맞춰
분포를 확인했다 (전체 19,370장, person 박스 439,046개 기준):

**박스 단위 분포**
| 구간 | 박스 수 | 비율 |
|---|---|---|
| <30% | 235,308 | 53.6% |
| 30~50% | 76,735 | 17.5% |
| 50~70% | 73,028 | 16.6% |
| ≥70% | 53,975 | 12.3% |

**이미지 단위 분포** (이미지 내 가장 심하게 가려진 박스 기준)
| 구간 | 이미지 수 | 비율 |
|---|---|---|
| <30% | 1,189 | 6.1% |
| 30~50% | 1,990 | 10.3% |
| 50~70% | 3,969 | 20.5% |
| ≥70% | 12,222 | 63.1% |

**occlusion 임계값별 서브셋 규모** (해당 비율 이상인 박스를 하나라도 포함하는 이미지 수):
- occlusion ≥30%: 18,181장 (93.9%) → `model/data/processed/crowdhuman/occ30_subset.txt`
- occlusion ≥50%: 16,191장 (83.6%) → `model/data/processed/crowdhuman/occ50_subset.txt`
- occlusion ≥70%: 12,222장 (63.1%) → `model/data/processed/crowdhuman/occ70_subset.txt`

CrowdHuman은 원래 밀집 군중 데이터셋이라 대부분 이미지에 심하게 가려진 사람이 최소 한 명은
있음 — 이미지 단위로는 occlusion 비율이 매우 높게 잡히지만, 실제 학습 시 유용한 신호는
박스 단위 분포(위 표)에 가깝다. 필요 시 위 서브셋 파일을 `split_dataset.py`의 커스텀 소스로
추가해 "심하게 가려진 사람" 비중을 높인 학습 서브셋을 구성할 수 있다.

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
  --variants-per-image 1
```

- 텍스처는 의사(pseudo) Perlin 노이즈 기반으로 4종(concrete/rebar/dust/tarp) 절차적 생성,
  상단 경계를 들쭉날쭉하게 처리해 잔해 더미처럼 보이도록 함
- 목표 비율은 이분 탐색으로 실제 픽셀 기준 occlusion 비율에 **±5% 이내로 수렴**시킴
  (수렴 실패 시 가장 근접한 결과 사용, 최대 8회 시도 — 처음엔 12회였으나 대규모 실행 속도를
  위해 8회로 축소, 이분탐색 특성상 8회면 이론적으로 1/256 정밀도라 품질 저하는 미미함)
- **라벨(full-extent bbox)은 원본 그대로 유지** — occlusion은 이미지에만 적용, 사람의 실제
  전체 범위 라벨은 변경하지 않음
- 모든 생성 기록은 `model/data/synthetic/metadata.csv`에 저장
  (`source_image, output_image, bbox_index, target_ratio, actual_ratio, texture_type`)
- 절차적 텍스처를 실제 촬영한 잔해 텍스처로 교체하면 합성 데이터 비율 상한을 60~70%까지
  완화 가능 (PRD 7.2절) — 향후 자체 촬영 진행 시 텍스처 이미지도 함께 확보해 반영 예정
- `--max-boxes-per-image`: CrowdHuman/WiderPerson처럼 이미지 하나에 사람이 매우 많은 경우
  (평균 19~32명, 최대 660명) 전원에게 occlusion을 적용하면 비현실적이고 처리 시간도 급증하므로,
  이미지당 최대 5개 박스만 무작위로 선택해 occlusion 적용 (나머지 박스는 라벨에는 그대로 포함,
  이미지만 원본 유지)
- `--limit`: 벤치마크/테스트용 상한 옵션 추가

### 3.1 실행 결과 (2026-07-15, COCO+CrowdHuman+WiderPerson 3개 소스 전체)

| 소스 | 입력 이미지 | 처리 속도 | 소요 시간 | 비고 |
|---|---|---|---|---|
| COCO | 66,808 | 43.9장/초 | 약 25분 | cap 없음 (평균 4.2박스/이미지로 밀도 낮음) |
| WiderPerson | 9,000 | 45.7장/초 | 약 3분 | `--max-boxes-per-image 5` |
| CrowdHuman | 19,370 | 7.8장/초 | 약 41분 | `--max-boxes-per-image 5` (박스 밀도가 높아 상대적으로 느림) |

**최종 결과**: `model/data/synthetic/`에 이미지 **95,175장** + 라벨 95,175개 + metadata.csv
413,839행(occlusion 처리한 박스 단위 기록). variants-per-image=1이지만 이미지 인덱스 기준으로
30/50/70% 비율이 순환 배분되어 전체적으로 고르게 섞임.

**중간에 발견한 이슈와 조치**:
- COCO 원본 이미지 중 3장(`000000142790.jpg` 등)이 다운로드 스크립트의 "존재하면 스킵" 로직
  때문에 손상된 채로 남아있었음 → 해당 3장 재다운로드로 해결, 재발 방지를 위해
  `download_coco_person_subset.py`가 다운로드 실패 시 부분 파일을 자동 삭제하도록 수정
- 위 3장 중 2장은 재다운로드 직후에도 augmentation 스크립트가 읽는 시점에 일시적으로
  다시 실패(십중팔구 백신 실시간 검사 등 환경적 요인) → `occlusion_augment.py`가 이미지 로드
  실패 시 크래시하지 않고 해당 이미지만 건너뛰도록 예외처리 추가. 최종적으로 66,808장 중
  66,805장 처리(3장 누락, 0.004% 수준으로 무시 가능)
- CrowdHuman/WiderPerson은 이미지 하나에 사람이 매우 많아(평균 19.4/32.3명) 전원에게
  occlusion을 적용하면 CrowdHuman 기준 초당 2장 수준으로 느려지고 결과도 비현실적이어서
  `--max-boxes-per-image 5`로 제한 (초당 2.1장 → 6.7장으로 개선)

**품질 확인**: 중간 크기 이상의 인물 bbox에서는 잔해 텍스처가 자연스럽게 하단부를 덮는 것을
샘플 이미지로 육안 확인함. 다만 CrowdHuman/COCO의 소형·원거리 인물 인스턴스는 bbox 자체가
작아 debris 패치도 상대적으로 작게 보이는 한계가 있음 — 실사용 시나리오(헬멧캠 근~중거리)와는
차이가 있으므로, 필요 시 향후 bbox 크기 기준 필터링을 추가하는 것을 고려할 수 있음.

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
- ~~CrowdHuman 실제 다운로드 및 배치~~ → **완료** (19,370장 확보·검증·YOLO 변환 및 occlusion
  서브셋 분석까지 완료, 2026-07-15)
- ~~WiderPerson 실제 다운로드 및 배치~~ → **완료** (9,000장 확보·검증·YOLO 변환 및 class 3
  부분가림 서브셋 분석까지 완료, 2026-07-15)
- OCHuman 신청 폼 제출 및 bbox 성격(amodal 여부) 원 논문 재확인
- 자체 촬영 일정 협의 (팀원 C, D) — SHOOTING_GUIDE.md 기준으로 진행
- 합성 텍스처를 실제 촬영 잔해 사진으로 교체하는 실험 (자체 촬영 진행 후)
- ~~COCO/CrowdHuman/WiderPerson에 `occlusion_augment.py` 적용해 합성 데이터 생성~~ →
  **완료** (95,175장 생성, 3장 결측 무시 가능 수준, 2026-07-15 — 상세는 3.1절)
- 자체 촬영 데이터까지 모두 갖춰지면 `split_dataset.py`로 전체(공개 95,178장 + 합성 95,175장
  + 자체촬영) train/val/test를 한 번에 분할하고, PRD 8장 스모크테스트 결과를 본 학습 규모로
  재검증 (사용자 결정: 자체 촬영 전까지 분할은 보류)
