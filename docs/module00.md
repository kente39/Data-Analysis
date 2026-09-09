# 0. 개발환경 셋업

Jupyter Notebook과 Google Colab, TensorFlow/Keras를 중심으로 환경을 잡습니다. 모바일 접근성을 우선한다면 Colab 위주로, 로컬 작업이 많다면 둘 다 준비해두는 걸 권장합니다.

## 0-1. Jupyter Notebook vs Google Colab

::: tip 핵심
빠른 실습·모바일·GPU는 **Colab**, 대용량 로컬 데이터·긴 실험은 **로컬 Jupyter**. 처음엔 Colab으로 "일단 돌아가게" 시작하는 걸 추천합니다.
:::

| 기준 | Jupyter (로컬) | Google Colab |
|---|---|---|
| 설치 | 직접 설치 필요 | 불필요 (브라우저만) |
| 모바일 | 사실상 불가 | 접속·실행 가능 |
| GPU | 본인 PC 사양, CUDA 설정 필요 | 무료 GPU 제공 |
| 데이터 | 로컬 파일 바로 접근 | 업로드 또는 Drive 마운트 |

## 0-2. Google Colab 시작하기

[colab.research.google.com](https://colab.research.google.com) 접속 → 새 노트 → 코드 셀에 입력 후 `Shift+Enter`. pandas·numpy·matplotlib·seaborn·scikit-learn·TensorFlow가 전부 기본 내장되어 있습니다.

::: warning 버전 드리프트 주의
Colab의 기본 라이브러리 버전은 시점에 따라 달라질 수 있습니다. 재현성이 중요하면 `!pip install tensorflow==2.x.x`처럼 버전을 명시하세요.
:::

GPU를 켜려면: `런타임 → 런타임 유형 변경 → 하드웨어 가속기 → GPU`

```python
# GPU가 실제로 잡혔는지 확인
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

내 데이터 불러오기:

```python
# 방법 1: 매 세션 업로드 (간단, 세션 끝나면 사라짐)
from google.colab import files
uploaded = files.upload()

# 방법 2: Google Drive 마운트 (영구 저장)
from google.colab import drive
drive.mount('/content/drive')
```

## 0-3. 로컬 Jupyter 환경 설치

::: warning Python 버전 주의
TensorFlow는 지원 Python 버전 범위가 정해져 있습니다(대체로 3.9~3.11). 너무 최신 Python에서는 아직 지원 안 되는 경우가 많으니, 공식 문서에서 지원 버전을 먼저 확인하세요.
:::

### Step 1~2. 가상환경 생성 (프로젝트별 분리 권장)

```bash
python -m venv myenv

# 활성화 (Windows)
myenv\Scripts\activate
# 활성화 (macOS / Linux)
source myenv/bin/activate
```

### Step 3. 라이브러리 설치

::: tip pip 대신 python -m pip 권장
컴퓨터에 Python이 여러 개면 그냥 `pip`는 어느 것의 pip인지 헷갈립니다. `python -m pip`는 "지금 이 Python 환경에 정확히" 설치합니다.
:::

```bash
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels jupyter
python -m pip install tensorflow
```

### Step 4. 실행

```bash
jupyter notebook
# 최근에는 jupyter lab (JupyterLab)도 널리 쓰임 — 파일 탐색·탭 관리가 더 편함
```

### Step 5. (자주 막히는 함정) 가상환경을 Jupyter 커널로 등록

::: warning 실전 최다 함정
터미널에선 설치 성공했는데 노트북에서 `import`가 안 되는 경우 — 설치한 가상환경과 노트북 커널이 다르기 때문입니다. A 공구함에 넣고 B 공구함으로 작업하는 상황이죠.
:::

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"
# 이후 Jupyter 우측 상단 커널 선택에서 "Python (myenv)" 선택
```

### Step 6. (선택) 환경 기록

```bash
python -m pip freeze > requirements.txt
# 나중에 재현: python -m pip install -r requirements.txt
```

## 0-4. TensorFlow 설치 함정

| 상황 | 문제 | 해결 |
|---|---|---|
| Apple Silicon (M칩) | 일반 설치로 GPU 안 잡힘 | `tensorflow-macos` + `tensorflow-metal` (최신 문서 확인) |
| GPU인데 CPU로만 동작 | CUDA/cuDNN 버전 불일치 | 정확한 버전 맞추거나, 번거로우면 Colab 사용 |
| Keras 별도 설치 충돌 | TF 2.x는 Keras 내장 | 따로 설치 말고 `tf.keras` 사용 |
| Windows 로컬 GPU | 버전/설치 제약 | WSL2 또는 Colab이 훨씬 간단 |

::: tip GPU가 항상 빠른 건 아닙니다
작은 데이터·간단한 모델은 GPU와 CPU 차이가 거의 없거나 오히려 CPU가 빠를 수 있습니다. GPU 효과는 대형 행렬 연산·딥러닝 학습에서 나타납니다.
:::

```python
import tensorflow as tf
print("TF 버전:", tf.__version__)
print("GPU 사용 가능:", tf.config.list_physical_devices('GPU'))
```

## ✅ 체크리스트

- Colab 또는 로컬 Jupyter 실행 환경 준비 완료
- `import tensorflow as tf; print(tf.__version__)` 에러 없이 출력
- (로컬이라면) 가상환경을 Jupyter 커널로 등록했다
- pandas/numpy/matplotlib/seaborn/scikit-learn/scipy/statsmodels import 성공
