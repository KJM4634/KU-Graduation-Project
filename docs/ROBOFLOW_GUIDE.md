# ROBOFLOW_GUIDE.md — 자체 촬영 프레임 Roboflow 라벨링 가이드

대상: `model/data/own_capture/subsampled_frames/` (및 업로드 편의를 위해 파일명을
`{영상명}_{프레임명}.jpg`로 평탄화해둔 `model/data/own_capture/subsampled_frames_flat/`)
**282장**. PRD 6장 원칙(전체범위 full-extent bbox)에 맞춰 Roboflow로 라벨링을 진행한다.

---

## 1. Roboflow 프로젝트 세팅

1. **계정 생성**: https://roboflow.com 에서 회원가입 (Google/GitHub 계정으로도 가능).
   무료 플랜(Public/Free)으로 이번 작업(282장)은 충분히 처리 가능.
2. **프로젝트 생성**: 대시보드에서 `Create New Project`
   - Project Name: 예) `dsar-assist-owncapture`
   - Project Type: **Object Detection**
   - Annotation Group(클래스 묶음 이름): 예) `person`
3. **이미지 업로드**: 프로젝트 생성 후 `Upload Data` 화면에서
   `model/data/own_capture/subsampled_frames_flat/` 폴더를 통째로 드래그 앤 드롭
   (평탄화된 폴더를 쓰는 이유: `subsampled_frames/`는 영상별 하위 폴더에 각각
   `frame_000000.jpg`부터 시작하는 동일한 파일명이 반복되어, 평탄 폴더 없이 올리면
   어느 영상의 프레임인지 구분하기 어려움 — `_flat` 폴더는 파일명 앞에 영상명을
   붙여둬서 이 문제가 없음)
   - 업로드 후 `Finish Uploading` → 배치 이름은 아무거나(예: `batch1`) 지정
4. **라벨링 클래스 설정**: 처음 어노테이션 화면에 들어가면 클래스를 물어보는데,
   **`person` 단일 클래스만** 등록한다 (PRD 6장: 잔해 매몰 여부와 무관하게 사람 하나만
   탐지하는 것이 목표이므로 클래스를 늘리지 않음). 클래스 이름은 정확히 `person`으로
   맞출 것 (나중에 export 시 다른 소스와 class index를 맞추기 위함, class 0 = person).
5. **라벨링 작업 배정** (팀원과 나눠서 할 경우): Roboflow 상단 `Assign` 메뉴에서
   이미지 배치를 팀원 계정에 나눠 할당 가능. 계정이 없다면 프로젝트에 `Invite` 후
   `Collaborator`로 초대.

---

## 2. 라벨링 원칙 (팀원 공유용 요약)

> 이 프로젝트의 라벨링 원칙은 딱 두 가지만 지키면 됩니다.

1. **전체범위(full-extent) bbox — 가려진 부분까지 포함해서 그리기**
   - 이불에 덮여 안 보이는 부분이 있어도, "실제로 사람의 몸이 있을 것으로 추정되는
     범위" 전체를 bbox로 그립니다. 예: 다리가 이불에 덮여 안 보여도, 이불 위로
     다리가 있을 것 같은 범위까지 박스를 넓혀서 그립니다.
   - 이유: 이 프로젝트는 "잔해에 부분적으로 가려진 사람"을 탐지하는 게 목표라,
     보이는 부분만 좁게 그리면 오히려 학습에 방해가 됩니다 (PRD 6장/8장에서 이미
     실험으로 확인된 내용).

2. **단, 카메라 프레임 밖으로 나간 부분은 추정해서 그리지 않기**
   - 예를 들어 발이 화면 바깥으로 잘려서 안 보이면, 그 발이 있을 위치까지
     상상해서 박스를 확장하지 않습니다. **박스는 이미지 경계선에서 멈춥니다.**
   - 즉, "이불에 가려짐"은 박스를 넓혀서 포함하고, "화면 밖으로 나감"은 이미지
     경계까지만 그린다 — 이 둘을 헷갈리지 않는 게 중요합니다.

3. (참고) 클래스는 `person` 하나만 사용, 한 이미지에 사람이 한 명이면 박스도 하나만
   그리면 됩니다.

---

## 3. Roboflow → 프로젝트 폴더로 가져오기

> ⚠️ 한 가지 확인이 필요합니다: split_dataset.py가 자체 촬영 라벨의 경로를
> **`model/data/own_capture/labeled/images`, `labels`**로 이미 하드코딩해뒀습니다
> (`model/scripts/split_dataset.py`의 `DEFAULT_SOURCES` 참고). 문의하신
> `model/data/processed/own_capture/` 대신, 기존 파이프라인과 바로 맞물리는
> **`model/data/own_capture/labeled/`**를 최종 목적지로 쓰는 걸 권장합니다. 아래
> 절차도 이 경로 기준으로 안내합니다 (혹시 정말 `processed/own_capture/`로
> 통일하고 싶으시면 말씀해주세요 — `split_dataset.py`의 경로 한 줄만 고치면 됩니다).

### 3.1 Roboflow에서 Export

1. 라벨링이 끝나면 프로젝트 좌측 메뉴 `Versions` → `Generate New Version`
2. Preprocessing/Augmentation 단계:
   - **Augmentation은 전부 끄기** (이미 `occlusion_augment.py`로 자체 augmentation
     파이프라인을 따로 운영 중이므로 중복 적용 방지)
   - Preprocessing도 `Auto-Orient` 정도만 켜고 Resize 등은 끄는 것을 권장
     (원본 해상도 유지, 리사이즈는 학습 시점에 처리)
3. `Generate` 완료 후 `Export Dataset` → Format: **YOLOv11** (또는 YOLOv8, 동일한
   normalized xywh txt 포맷이라 호환됨) 선택
4. `download zip` 선택해 로컬로 받기

### 3.2 로컬 폴더로 가져오기

Roboflow YOLO export는 기본적으로 `train/`, `valid/`, `test/` 로 이미 분할해서 주는데,
이 프로젝트는 **소스별로 자체 분할 비율(own_capture는 30/20/50)을 `split_dataset.py`가
따로 적용**하므로, Roboflow가 나눠준 분할은 무시하고 이미지+라벨을 전부 합쳐서 넣는다.

```bash
# 압축 해제한 Roboflow export 폴더 구조 예시:
#   roboflow_export/train/images, train/labels
#   roboflow_export/valid/images, valid/labels
#   roboflow_export/test/images,  test/labels
# (data.yaml, README.roboflow.txt 등은 무시)

mkdir -p model/data/own_capture/labeled/images model/data/own_capture/labeled/labels

for split in train valid test; do
  cp roboflow_export/$split/images/* model/data/own_capture/labeled/images/
  cp roboflow_export/$split/labels/* model/data/own_capture/labeled/labels/
done
```

(Windows PowerShell 기준)
```powershell
New-Item -ItemType Directory -Force model/data/own_capture/labeled/images, model/data/own_capture/labeled/labels
foreach ($split in "train","valid","test") {
    Copy-Item "roboflow_export/$split/images/*" model/data/own_capture/labeled/images/
    Copy-Item "roboflow_export/$split/labels/*" model/data/own_capture/labeled/labels/
}
```

### 3.3 확인

```bash
# 이미지 수 = 라벨 수 = 282 (전부 person 1개씩 라벨링됐다면)
ls model/data/own_capture/labeled/images | wc -l
ls model/data/own_capture/labeled/labels | wc -l

# 라벨 클래스가 전부 0(person)인지 확인
cut -d' ' -f1 model/data/own_capture/labeled/labels/*.txt | sort -u
```

이후 모든 공개 데이터셋(COCO/CrowdHuman/WiderPerson)이 준비되면, `split_dataset.py`가
`own_capture/labeled/images`를 자동으로 인식해 30/20/50 비율로 train/val/test에
포함시킨다 (DATASET.md 5장 참고).
