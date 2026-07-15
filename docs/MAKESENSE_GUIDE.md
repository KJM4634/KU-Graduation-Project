# MAKESENSE_GUIDE.md — 자체 촬영 프레임 makesense.ai 라벨링 가이드

대상: `model/data/own_capture/subsampled_frames_flat/` 282장.

LabelImg가 이 PC에서 PyQt6/PyQt5 호환성 문제로 이미지 로딩에 계속 실패해
(`docs/LABELIMG_GUIDE.md` 참고, 폐기됨), **[makesense.ai](https://www.makesense.ai/)**로
전환한다. 웹 브라우저에서 여는 도구이지만, 공식 저장소(SkalskiP/make-sense) 설명에 따르면
**"We don't store your images, because we don't send them anywhere in the first place"** —
즉 이미지를 서버로 전송하지 않고 **브라우저 안에서만 처리**하므로, LabelImg를 쓰려던
이유(오프라인/프라이버시)와 실질적으로 동일한 조건을 만족한다.

---

## 1. 사용 방법

1. 브라우저(Chrome/Edge 등)에서 https://www.makesense.ai/ 접속
2. `Get Started` 클릭
3. 이미지 추가: `model/data/own_capture/subsampled_frames_flat/` 폴더 안의 이미지
   282장을 전부 드래그 앤 드롭 (탐색기에서 폴더 열어 전체 선택 → 드래그)
4. 작업 종류 선택: **`Object Detection`**
5. 라벨 목록 설정 화면에서 **`person`** 하나만 추가 (Create Labels List 수동 입력,
   또는 `model/data/own_capture/classes.txt`를 만들어뒀다면 그 파일을 업로드해도 됨 —
   내용은 `person` 한 줄)
6. `Start project` → 라벨링 에디터 화면으로 진입

---

## 2. 라벨링 원칙 (기존 가이드와 동일)

1. **전체범위(full-extent) bbox** — 이불에 덮여 안 보이는 부분도 사람이 있을 것으로
   추정되는 범위까지 박스를 그린다.
2. **화면 밖으로 나간 부분은 추정해서 그리지 않는다** — 박스는 이미지 경계에서 멈춘다.
3. 클래스는 `person` 하나만 사용, 이미지당 사람이 한 명이면 박스도 하나만.

---

## 3. 조작법

- 좌측 툴바에서 사각형(bounding box) 도구를 선택한 뒤, 이미지 위에서 클릭-드래그로 박스를 그림
- 박스를 그리면 라벨 선택 팝업이 뜨는데 `person` 선택 (라벨이 하나뿐이면 자동 지정되는
  경우도 있음)
- **키보드 단축키** (공식 위키 기준):
  | 단축키 | 동작 |
  |---|---|
  | `Ctrl`(Win)/`⌥`(Mac) + `←`/`→` | 이전/다음 이미지 |
  | `Ctrl`/`⌥` + `0~9` | 해당 번호의 라벨 선택 |
  | `Delete`(Win)/`Backspace`(Mac) | 선택된 박스 삭제 |
  | `Ctrl`/`⌥` + `+`/`-` | 확대/축소 |
- 화면 우측(또는 좌측) 썸네일 목록에서 작업한 이미지는 체크 표시가 되어 진행 상황 확인 가능
- 282장 전부 작업 후 상단/좌측 `Actions` 메뉴로 이동

---

## 4. Export

1. `Actions` → **`Export Labels`**
2. 포맷 선택: **`A .zip package containing files in YOLO format`**
   - 좌표 형식은 표준 YOLO와 동일: `label_index xc yc w h` (전부 0~1 정규화 값) —
     우리 프로젝트 포맷과 그대로 호환됨 (class 0 = person)
3. zip 다운로드 후 압축 해제

> **참고**: zip 내부의 정확한 파일 구성(라벨별 txt 파일명 규칙, 클래스 목록 파일 유무 등)은
> makesense.ai 공식 문서에 상세히 명시돼 있지 않다. 압축을 풀어보면 각 이미지 파일명과
> 동일한 이름의 `.txt` 라벨 파일들이 나오는 것이 일반적이므로, 아래 3.2 절차 전에 압축
> 해제한 폴더를 한 번 열어 실제 파일 구성을 확인한 뒤 진행할 것.

### 4.1 로컬 폴더로 가져오기

압축 해제한 폴더에 `.txt` 라벨 파일들만 있다면(이미지는 재포함되지 않는 경우가 일반적 —
어차피 원본은 `subsampled_frames_flat/`에 이미 있음):

```powershell
New-Item -ItemType Directory -Force model/data/own_capture/labeled/images, model/data/own_capture/labeled/labels
Copy-Item model/data/own_capture/subsampled_frames_flat/*.jpg model/data/own_capture/labeled/images/
Copy-Item "<압축해제폴더>\*.txt" model/data/own_capture/labeled/labels/
```

(클래스 목록 파일(`classes.txt`/`labels.txt` 등)이 zip에 포함돼 있다면 그건
`labeled/labels/`에 넣지 말고 별도 보관 — YOLO 학습 파이프라인은 `model/data/dataset.yaml`의
`names: ["person"]`로 클래스를 이미 정의해뒀으므로 불필요)

### 4.2 확인

```bash
ls model/data/own_capture/labeled/images | wc -l   # 282
ls model/data/own_capture/labeled/labels | wc -l   # 282

# 라벨 클래스가 전부 0(person)인지, 좌표가 0~1 범위인지 샘플 확인
head -3 model/data/own_capture/labeled/labels/*.txt | head -20
```

이후 `model/scripts/split_dataset.py`가 `own_capture/labeled/images`를 자동으로 인식해
30/20/50 비율로 train/val/test에 포함시킨다 (DATASET.md 5장 참고).
