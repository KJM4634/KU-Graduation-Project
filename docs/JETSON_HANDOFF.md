# JETSON_HANDOFF.md — Jetson Orin Nano 배포 인수인계 가이드

이 문서는 Jetson Orin Nano를 **처음 만져보는 사람**이 혼자 보고 따라할 수 있도록 작성했다.
모델 학습(RTX 3090 PC)과 엣지 코드(`edge/`)는 이미 완료돼 있고, 이 문서는 그 결과물을
Jetson에 올려서 **실제로 카메라 4개 + TensorRT 추론이 동작하는 것까지** 확인하는 절차다.

막히면 마지막 "6. 막혔을 때 체크리스트"부터 봐도 된다. 순서대로 하나씩 실행하면서
"정상 결과 예시"와 실제 화면이 비슷한지 대조하면 된다.

---

## 0. 시작 전에 알아야 할 것

- Jetson은 Windows PC와 CPU 구조(ARM)가 달라서, **PC에서 쓰던 `requirements.txt`를
  Jetson에서 그대로 `pip install`하면 안 된다.** (2번에서 자세히 설명)
- 이 문서에서 `$` 로 시작하는 줄은 Jetson의 터미널(Terminal)에 그대로 복사해서 붙여넣고
  Enter를 누르면 되는 명령어다.
- Jetson 전원을 켜고 화면이 뜬 상태(모니터/키보드 연결 또는 SSH 접속)에서 시작한다.

---

## 1. 전달할 파일 목록 및 받는 방법

### 1.1 Git 저장소로 받는 것 (코드, 문서)

아래는 모두 GitHub 저장소에 포함되어 있어 `git clone` 한 번으로 받아진다.

| 항목 | 위치 |
|---|---|
| 엣지 코드 전체 | `edge/` (camera, gps, notify, comm, config) |
| 요구사항 문서 | `docs/PRD.md`, `docs/STACK.md`, `docs/MODEL_REPORT.md` |
| 패키지 목록(참고용, 2번에서 설명) | `requirements.txt` |

```bash
git clone https://github.com/KJM4634/KU-Graduation-Project.git
cd KU-Graduation-Project
```

### 1.2 별도로 받아야 하는 것 (모델 가중치 — Git에 없음)

아래 두 파일은 용량 문제로 `.gitignore`에 의해 **Git 저장소에는 포함되어 있지 않다.**
반드시 Google Drive 또는 USB로 별도 전달받아야 한다.

| 파일명 | 용량 | 용도 |
|---|---|---|
| `dsar_n_full.pt` | 약 5.3MB | (참고용, TensorRT 변환에는 아래 onnx를 씀) |
| `dsar_n_full.onnx` | 약 11MB | **이 파일을 Jetson에서 TensorRT로 변환한다** |

**전달 방법 (택1)**
- **Google Drive**: [팀원 B가 업로드 후 이 자리에 실제 공유 링크를 채워넣을 것] →
  다운로드 후 저장소를 클론한 폴더 기준 `model/weights/` 안에 넣는다.
- **USB**: 위 2개 파일을 USB에 담아 Jetson에 직접 복사. 마찬가지로 `model/weights/`
  폴더에 넣는다 (폴더가 없으면 `mkdir -p model/weights` 로 먼저 만든다).

받은 뒤 아래 명령으로 두 파일이 제자리에 있는지 확인한다.

```bash
ls -la model/weights/dsar_n_full.pt model/weights/dsar_n_full.onnx
```

**정상 결과 예시**
```
-rw-r--r-- 1 user user  5613764 Jul 17 00:45 model/weights/dsar_n_full.pt
-rw-r--r-- 1 user user 10604084 Jul 17 00:45 model/weights/dsar_n_full.onnx
```
(파일이 없다고 나오면 → "6.3 그 외 자주 겪는 문제" 참고)

---

## 2. Jetson에서 실행할 순서

### 2단계-0. JetPack 버전 확인

Jetson은 JetPack이라는 이름의 패키지(리눅스 + CUDA + cuDNN + TensorRT 묶음)가 이미
설치돼 있다. 버전을 확인해야 이후 설치할 패키지 버전을 맞출 수 있다.

```bash
cat /etc/nv_tegra_release
sudo apt-cache show nvidia-jetpack | grep -E "^Package|^Version"
```

**정상 결과 예시**
```
# R36 (release), REVISION: 3.0, GCID: ...
```
```
Package: nvidia-jetpack
Version: 6.0+b106
```
→ `R36`, `6.0` 같은 숫자가 이 Jetson의 JetPack 버전이다. **이 숫자를 메모해두고 이후
단계(특히 2단계-2 torch 설치)에서 사용한다.** (모르면 넘어가지 말고 아래 "6.2" 참고)

TensorRT가 이미 설치돼 있는지도 함께 확인한다.

```bash
dpkg -l | grep -i tensorrt
ls /usr/src/tensorrt/bin/trtexec
```

**정상 결과 예시**: `tensorrt` 관련 패키지 목록이 여러 줄 나오고, 마지막 줄에서
`/usr/src/tensorrt/bin/trtexec` 경로가 그대로 출력되면 정상 (파일이 실제로 존재한다는 뜻).

---

### 2단계-1. 저장소 클론 + 가중치 배치

1.1, 1.2에서 이미 했다면 건너뛴다. 안 했다면 지금 한다.

```bash
git clone https://github.com/KJM4634/KU-Graduation-Project.git
cd KU-Graduation-Project
mkdir -p model/weights
# (Drive에서 받은 파일을 model/weights/ 로 이동, 또는 USB에서 복사)
ls model/weights/dsar_n_full.onnx
```

---

### 2단계-2. 필요 패키지 설치

**절대 이렇게 하지 말 것**:
```bash
pip install -r requirements.txt   # X — Windows/x86용 torch wheel이라 Jetson(ARM)에서 설치 자체가 실패함
```

`requirements.txt`의 `torch==2.13.0+cu130` 같은 항목은 Windows RTX 3090 PC 전용이다.
Jetson(ARM64 + JetPack 내장 CUDA)에서는 **JetPack 버전에 맞는 NVIDIA 전용 wheel**을
따로 설치해야 한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

torch/torchvision은 공식 Ultralytics NVIDIA Jetson 가이드를 그대로 따라 설치한다
(2단계-0에서 확인한 JetPack 버전에 맞는 설치 명령이 문서에 정리되어 있다):

**https://docs.ultralytics.com/guides/nvidia-jetson/**

위 문서의 "Install Ultralytics" 섹션에 나오는, 본인 JetPack 버전(R36 / 6.x 등)에 해당하는
torch/torchvision 설치 명령을 그대로 실행한 뒤, 나머지 패키지를 설치한다.

```bash
pip install ultralytics onnxruntime-gpu pyserial pynmea2
```

**설치가 끝나면 확인:**

```bash
python3 -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

**정상 결과 예시**
```
torch: 2.3.0
CUDA available: True
```
`CUDA available: False`가 나오면 → "6.2 TensorRT/torch 관련 문제" 참고.

---

### 2단계-3. ONNX → TensorRT 변환

`trtexec`(TensorRT에 기본 포함된 변환 도구, 2단계-0에서 경로 확인함)를 사용한다.
Python/torch 설치와 무관하게 동작하므로, 2단계-2가 아직 불안하더라도 이 단계는
독립적으로 먼저 시도해볼 수 있다.

```bash
cd model/weights
/usr/src/tensorrt/bin/trtexec --onnx=dsar_n_full.onnx --saveEngine=dsar_n_full.engine --fp16
```

변환에는 수 분 정도 걸릴 수 있다. **정상 완료 시 맨 마지막 줄에 이렇게 나온다:**

```
[I] === Performance summary ===
[I] Throughput: 123.45 qps
[I] ...
&&&& PASSED TensorRT.trtexec [TensorRT v8xxx] # /usr/src/tensorrt/bin/trtexec --onnx=dsar_n_full.onnx --saveEngine=dsar_n_full.engine --fp16
```

핵심은 맨 끝의 **`&&&& PASSED`** 문구다. 이게 안 보이고 `FAILED`가 보이거나 중간에
에러로 멈추면 → "6.2 TensorRT 변환 실패" 참고.

완료 후 엔진 파일이 생겼는지 확인:

```bash
ls -la dsar_n_full.engine
cd ../..
```

---

### 2단계-4. 카메라 4개 인식 확인

USB 카메라 4개를 Jetson에 모두 연결한 뒤 확인한다.

```bash
ls /dev/video*
```

**정상 결과 예시** (카메라 1대당 보통 index 2개씩 잡히는 경우도 있어 숫자가 8개까지
나올 수도 있음 — 아래 "6.1"에서 구분법 설명):
```
/dev/video0  /dev/video1  /dev/video2  /dev/video3
```

각 카메라가 실제로 어떤 장치인지 이름까지 확인하려면:

```bash
sudo apt install -y v4l-utils   # 없으면 설치
v4l2-ctl --list-devices
```

**정상 결과 예시**
```
USB Camera (usb-70090000.xusb-2.1):
	/dev/video0
	/dev/video1

USB Camera (usb-70090000.xusb-2.2):
	/dev/video2
	/dev/video3
...
```
→ 카메라 1대당 `/dev/video0`, `/dev/video1` 처럼 2개가 묶여 나오면, **더 작은 번호가
실제 캡처용 index**인 경우가 많다 (나머지는 메타데이터용). 이 문서의 다음 단계에서
쓸 index 4개(전/후/좌/우 담당)를 이 출력에서 확인해 메모해둔다.

인식이 안 되면 → "6.1 카메라 인식이 안 될 때" 참고.

---

### 2단계-5. 4채널 파이프라인 실행

`edge/camera/inference_pipeline.py`를 실행한다. `--source cameras`와 함께 2단계-4에서
확인한 index 4개를 방향별로 지정한다 (아래는 전형적인 index 0/2/4/6 예시 — 실제로는
본인 환경에서 확인한 숫자로 바꿔서 사용).

```bash
cd edge/camera
python3 inference_pipeline.py \
  --source cameras \
  --cam-front 0 --cam-back 2 --cam-left 4 --cam-right 6 \
  --model ../../model/weights/dsar_n_full.engine \
  --conf 0.25 \
  --iterations 0 \
  --show
```

- `--iterations 0`은 무한 반복(Ctrl+C 또는 화면에서 `q` 키로 종료)이다. 처음 테스트할
  때는 `--iterations 20` 정도로 짧게 끊어서 확인하는 것을 권장한다.
- `--show`를 주면 방향별로 창 4개(front/back/left/right)가 뜨고, 사람이 있으면
  **초록색 사각형 박스 + confidence 숫자**가 실시간으로 그려진다.

**정상 동작 확인 방법**
1. 터미널에 아래처럼 프레임마다 방향별 탐지 요약이 출력된다.
   ```
   --- frame 0 ---
     front: 탐지 없음
      back: 1건 탐지 (최고 conf=0.81)
      left: 탐지 없음
     right: 2건 탐지 (최고 conf=0.63)
   ```
2. 동시에 화면에 4개 창이 뜨고, 사람이 카메라 앞을 지나가면 해당 방향 창에만
   박스가 그려지고 **다른 방향 창에는 그려지지 않는지** 확인한다 (4채널이 서로
   독립적으로 동작한다는 핵심 검증 포인트).
3. `q` 키를 누르면 창이 닫히고 프로그램이 정상 종료된다.

여기까지 확인되면 Jetson 배포 파이프라인의 핵심 동작 검증은 끝난 것이다.

---

## 3. STACK.md 기준 버전 호환성 주의사항

RTX 3090 PC(`docs/STACK.md`)와 Jetson은 아래처럼 **의도적으로 다른 패키지 구성**을 쓴다.
"버전이 다르다고 잘못된 게 아니다" — 아래 표 기준으로만 맞으면 정상이다.

| 항목 | RTX 3090 PC | Jetson | 주의사항 |
|---|---|---|---|
| torch | 2.13.0+cu130 (PyPI cu130 wheel) | JetPack 버전별 NVIDIA 전용 wheel | PC용 wheel을 Jetson에 설치 시도하면 `ERROR: Could not find a version that satisfies the requirement torch==2.13.0+cu130` 로 즉시 실패함 (ARM 아키텍처용 빌드가 존재하지 않기 때문) |
| 추론 방식 | PyTorch(.pt) 또는 ONNX(onnxruntime-gpu) | **TensorRT(.engine)** | PRD 6장에 명시된 배포 조합(YOLO11→ONNX→TensorRT FP16)을 따름. Jetson에서 `.pt`로 직접 추론도 가능은 하지만 훨씬 느리므로 데모/배포에는 반드시 `.engine` 사용 |
| ultralytics | 8.4.95 (PC에서 검증) | 같은 버전대를 권장하되, JetPack 6.x + 최신 torch 조합에서는 ultralytics가 자동으로 더 최신 버전을 요구할 수 있음 | `pip install ultralytics`가 8.4.95보다 최신 버전을 설치해도 정상 — TensorRT `.engine` 로딩은 버전 간 호환성이 넓은 편 |
| opencv-python | 5.0.0.93 (pip wheel) | JetPack에 이미 OpenCV가 시스템 설치돼 있는 경우가 많음 | `pip install opencv-python`을 venv에 또 설치하면 시스템 OpenCV(CUDA/GStreamer 가속 포함)와 충돌 가능. `python3 -c "import cv2; print(cv2.__version__)"`로 먼저 확인 후, 이미 있으면 pip 설치를 생략하고 시스템 사이트 패키지를 쓰는 것을 권장 (`python3 -m venv --system-site-packages .venv`로 venv를 다시 만들면 됨) |
| onnxruntime-gpu | 1.27.0 (PyPI) | Jetson 전용 빌드 필요 (일반 PyPI onnxruntime-gpu는 ARM 미지원) | 이 프로젝트의 Jetson 배포 경로는 onnxruntime이 아니라 **TensorRT**이므로, onnxruntime-gpu는 Jetson에서는 설치하지 않아도 된다 (2단계-2에서 생략 가능) |

---

## 4. 막혔을 때 체크리스트

### 4.1 카메라 인식이 안 될 때

- [ ] `ls /dev/video*` 자체에 아무것도 안 뜨는가? → USB 케이블/포트 문제. 다른 USB
      포트로 꽂아보고, `dmesg | tail -30`으로 카메라 연결 시 커널 로그가 잡히는지 확인.
- [ ] `dmesg | tail -30`에 `usb 1-2: device descriptor read/64, error -71` 같은
      메시지가 보이는가? → 전력 부족 가능성. Jetson USB 허브에 4개를 한번에 물리면
      전력이 부족할 수 있어, **전원 공급형(powered) USB 허브** 사용을 검토.
- [ ] index는 잡히는데 `cv2.VideoCapture(index)`가 `isOpened() == False`를 반환하는가?
      → 권한 문제일 수 있음. 아래로 현재 사용자를 `video` 그룹에 추가 후 **재로그인**.
      ```bash
      sudo usermod -aG video $USER
      ```
- [ ] 4개 중 일부만 열리는가? → 동시에 여는 개수 제한(대역폭/드라이버 이슈)일 수
      있음. `v4l2-ctl --list-devices`에서 확인한 index가 실제로 4개의 **서로 다른
      물리 카메라**를 가리키는지 다시 확인 (같은 카메라의 metadata용 index를 중복
      지정했을 가능성).
- [ ] 화면(`--show`)은 안 뜨고 터미널만 도는가? → SSH로 접속해서 실행 중이라면
      디스플레이가 없는 게 정상 원인. 모니터가 직결된 상태에서 로컬 터미널로
      실행하거나, SSH라면 `ssh -X`(X11 forwarding)로 접속했는지 확인.

### 4.2 TensorRT 변환(trtexec) 실패할 때

- [ ] `[E] [TRT] ... Your ONNX model has been generated with INT64 weights` 같은
      경고만 뜨고 계속 진행되는가? → 경고(Warning)이지 실패가 아니다. 맨 마지막에
      `&&&& PASSED`가 뜨면 정상 진행된 것이니 무시해도 된다.
- [ ] `Segmentation fault` 또는 `Killed`로 중간에 죽는가? → 메모리 부족 가능성.
      다른 프로그램(특히 브라우저, 다른 추론 프로세스)을 모두 종료하고 재시도.
      `--fp16` 옵션이 이미 메모리를 절반으로 줄여주므로, 그래도 실패하면
      `free -h`로 여유 메모리를 확인.
- [ ] `[E] [TRT] onnx::ParseError` 류의 파싱 에러가 뜨는가? → onnx 파일 자체가
      손상되었거나 전달 과정에서 잘렸을 가능성. `ls -la dsar_n_full.onnx`로 용량이
      약 11MB(10604084 byte)가 맞는지 다시 확인, 다르면 1.2번부터 재전달.
- [ ] `command not found: trtexec` 인가? → 경로가 다른 것. 2단계-0에서 확인한
      `/usr/src/tensorrt/bin/trtexec` 전체 경로를 그대로 써야 한다 (PATH에 안 잡혀
      있는 JetPack 버전이 있음).
- [ ] 변환은 성공했는데 `dsar_n_full.engine` 파일이 매우 작거나(수십 KB 이하)
      비정상적으로 보이는가? → 변환 도중 로그를 위로 스크롤해서 실제로 `&&&& PASSED`
      가 있었는지 재확인. 없었다면 실패한 것이므로 재시도.

### 4.3 그 외 자주 겪는 문제

| 에러 메시지 | 원인 | 대처 |
|---|---|---|
| `FileNotFoundError: model/weights/dsar_n_full.pt` (또는 `.onnx`, `.engine`) | 1.2에서 가중치 파일을 못 받았거나, 실행 시 상대경로가 안 맞음 (특히 `edge/camera` 폴더 안에서 실행 중이면 `../../model/weights/...`처럼 상위로 두 칸 올라가야 함) | `ls model/weights/`로 파일이 실제로 있는지, 실행 중인 폴더가 어디인지(`pwd`) 확인 |
| `ModuleNotFoundError: No module named 'ultralytics'` (또는 cv2, torch 등) | venv를 활성화하지 않고 실행함 | `source .venv/bin/activate` 후 다시 실행 (터미널을 새로 열면 매번 다시 activate 필요) |
| `torch.cuda.is_available()`가 `False` | JetPack 버전에 안 맞는 torch를 설치했거나, PC용 pip 기본 torch가 설치됨 | `pip uninstall torch torchvision`으로 지운 뒤, 2단계-2의 Ultralytics 공식 가이드에서 **본인 JetPack 버전에 맞는** 설치 명령으로 재설치 |
| 창은 뜨는데 화면이 매우 느리거나(수 초에 1프레임) 멈춘 듯 보임 | `.pt`나 `.onnx`로 추론 중이라 TensorRT 가속을 안 받고 있음 | `--model` 인자가 `dsar_n_full.engine`(TensorRT)을 가리키는지 확인. `.pt`를 그대로 쓰고 있다면 2단계-3을 다시 실행해 엔진 파일부터 만들 것 |
| 부저/LCD 관련 코드(`edge/notify/`)가 동작 안 함 | 정상 — GPIO/LCD 하드웨어 연결부는 아직 미구현 상태 (의도적으로 판단 로직만 분리 구현됨, `docs/MODEL_REPORT.md` 및 `edge/notify/decision.py` 참고) | Jetson에 부저/LCD를 실제로 연결한 뒤 하드웨어 제어 코드를 추가로 작성해야 하는 단계이며, 이 문서 범위 밖 |

---

## 5. 다음에 할 일 (이 문서 이후)

- GPS 모듈(UART)을 실제로 연결하면 `edge/gps/simulator.py`의 `parse_nmea_line()` 함수를
  그대로 재사용해 실기 GPS 파싱이 가능하다 (시뮬레이터와 파싱 로직이 동일하게 설계됨).
- 부저(GPIO)/LCD 하드웨어를 연결하면 `edge/notify/decision.py`의 `decide()` 판단 결과를
  받아 실제 GPIO 핀 제어 코드를 추가하면 된다 (판단 로직과 하드웨어 제어가 분리돼 있어
  이 파일은 수정할 필요 없음).
- Jetson 실측 FPS/발열/전력을 확인해 `docs/MODEL_REPORT.md`에 추가 기록 (PRD 4.1절,
  11.2절 검증 지표 참고).
