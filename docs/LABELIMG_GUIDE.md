# LABELIMG_GUIDE.md — 자체 촬영 프레임 오프라인 라벨링 가이드

대상: `model/data/own_capture/subsampled_frames/` (또는 `subsampled_frames_flat/`) 282장.
Roboflow(클라우드 업로드) 대신 **완전히 로컬에서 동작하는 LabelImg**로 라벨링한다.

---

## 1. 설치

이 프로젝트 가상환경(`.venv`)에 설치한다. LabelImg는 PyQt5 기반인데, 프로젝트에 이미
설치된 PyQt6(엣지 LCD용, `edge/notify` 참고)와는 별개 패키지라 같은 가상환경에 함께
설치해도 충돌하지 않는다 (실제로 별도 테스트 환경에서 설치·의존성 해석까지 확인함).

```
.venv\Scripts\pip install labelImg
```

설치 후 `.venv\Scripts\labelImg.exe`가 생성된다.

> 참고: LabelImg는 몇 년째 큰 업데이트가 없는 프로젝트지만, bbox를 그려서 YOLO 포맷으로
> 저장하는 기본 기능은 안정적으로 동작한다. 이 작업(단일 클래스, 이미지당 박스 1개)에는
> 충분하다.

---

## 2. 클래스 파일 준비

라벨링 시작 전, `person` 한 줄만 있는 클래스 파일을 만든다 (class 0 = person, 프로젝트
전체 라벨링 규칙과 동일).

```
model/data/own_capture/classes.txt
```
내용:
```
person
```

---

## 3. 실행

```
.venv\Scripts\labelImg.exe model/data/own_capture/subsampled_frames_flat model/data/own_capture/classes.txt
```

- 첫 번째 인자: 라벨링할 이미지 폴더
- 두 번째 인자: 위에서 만든 클래스 파일 (미리 지정해두면 실행 시 클래스 목록이 자동으로 뜸)

실행 후 프로그램이 열리면:
1. 좌측 메뉴에서 **저장 형식을 YOLO로 변경**: 왼쪽 세로 툴바에 `PascalVOC`라고 써진 버튼을
   클릭하면 `YOLO`로 바뀜 (클릭할 때마다 PascalVOC ↔ YOLO ↔ CreateML 순환). 반드시 `YOLO`로
   맞춰야 우리 프로젝트 포맷과 바로 호환됨.
2. `Change Save Dir` 버튼으로 라벨(.txt) 저장 위치를 지정 (예: 이미지와 같은 폴더에 저장해도
   되고, 별도 폴더로 지정해도 됨 — 아래 4번에서 다시 정리)

---

## 4. 라벨링 작업 (단축키)

| 키 | 동작 |
|---|---|
| `w` | 새 박스 그리기 시작 |
| `d` | 다음 이미지 |
| `a` | 이전 이미지 |
| `del` | 선택된 박스 삭제 |
| `Ctrl+S` | 저장 |

**작업 순서**: `w` → 마우스로 사람 전체범위(가려진 부분 포함) bbox 드래그 → 클래스 선택
팝업에서 `person` 선택 → `Ctrl+S` → `d`로 다음 이미지. 282장 전부 반복.

**라벨링 원칙** (Roboflow 가이드와 동일):
1. **전체범위(full-extent) bbox** — 이불에 덮여 안 보이는 부분도 사람이 있을 것으로
   추정되는 범위까지 박스를 그린다.
2. **화면 밖으로 나간 부분은 추정해서 그리지 않는다** — 박스는 이미지 경계에서 멈춘다.
3. 클래스는 `person` 하나만 사용, 이미지당 사람이 한 명이면 박스도 하나만.

---

## 5. 결과물 정리

LabelImg는 이미지 파일명과 동일한 이름의 `.txt`(YOLO 포맷: `class_id xc yc w h` 정규화
좌표)를 저장한다. 라벨링이 끝나면 프로젝트 관례에 맞춰 이미지와 라벨을 분리해서 배치한다.

```powershell
New-Item -ItemType Directory -Force model/data/own_capture/labeled/images, model/data/own_capture/labeled/labels
Copy-Item model/data/own_capture/subsampled_frames_flat/*.jpg model/data/own_capture/labeled/images/
Copy-Item model/data/own_capture/subsampled_frames_flat/*.txt model/data/own_capture/labeled/labels/
```

(라벨 저장 위치를 `Change Save Dir`로 아예 별도 폴더로 지정해뒀다면, 그 폴더 내용을
`labeled/labels/`로 옮기면 됨)

**확인**:
```bash
ls model/data/own_capture/labeled/images | wc -l   # 282
ls model/data/own_capture/labeled/labels | wc -l   # 282 (classes.txt 제외)
```

이후 `model/scripts/split_dataset.py`가 `own_capture/labeled/images`를 자동으로 인식해
30/20/50 비율로 train/val/test에 포함시킨다 (DATASET.md 5장 참고).
