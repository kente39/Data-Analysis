# 7. 딥러닝 — TensorFlow / Keras

Module 6까지는 scikit-learn으로 머신러닝을 다뤘습니다. 이번 모듈에서는 **딥러닝(신경망)** 으로 같은 두 문제 — 회귀(`height_cm` 예측)와 분류(`is_blooming` 예측) — 를 풀어 보고, **6장의 머신러닝 결과와 직접 비교**합니다.

핵심 질문은 하나입니다: **"딥러닝이 항상 더 좋은가?"** 결론부터 말하면 이 데이터에서는 그렇지 않습니다. 그 이유를 개념부터 차근차근 확인합니다.

## 7-1. 딥러닝 기본 개념

::: tip 딥러닝이란?
딥러닝은 **인공신경망(neural network)** 을 여러 층 쌓아 학습하는 머신러닝의 한 갈래입니다. 특히 **층(layer)이 깊은(deep)** 신경망을 쓴다고 해서 '딥'러닝입니다.

Module 6의 모델들이 사람이 정한 구조(선형식, 트리 분할)로 학습했다면, 신경망은 **여러 층의 가중치를 스스로 조정**하며 입력과 출력의 관계를 근사합니다.
:::

![층을 깊게 쌓은 심층 신경망 개요](./imgs/module07/m7_dnn.png)

### 신경망의 기본 단위 — 뉴런과 층

가장 작은 단위는 **뉴런(퍼셉트론)** 입니다. 뉴런 하나는 이렇게 계산합니다.

1. 각 입력에 **가중치(weight)** 를 곱해 모두 더하고, **편향(bias)** 을 더합니다. (선형 결합)
2. 그 값을 **활성화 함수**에 통과시켜 출력합니다.

![뉴런(퍼셉트론)의 계산 과정](./imgs/module07/m7_neuron.png)

이 뉴런을 나란히 모은 것이 **층(layer)** 이고, 층을 여러 개 쌓은 것이 신경망입니다.

- **입력층** — 전처리된 특성이 들어오는 곳
- **은닉층(hidden layer)** — 중간에서 특징을 조합하는 층. 여기가 깊어질수록 '딥'
- **출력층** — 최종 예측. 회귀는 숫자 1개, 이진분류는 확률 1개

![입력층-은닉층-출력층 구조](./imgs/module07/m7_layers.png)

### 활성화 함수 — 왜 필요한가

활성화 함수가 없으면 층을 아무리 쌓아도 결국 **하나의 선형식**과 같아져 복잡한 관계를 학습할 수 없습니다. 활성화 함수는 **비선형성**을 넣어 신경망이 곡선 같은 복잡한 패턴도 표현하게 해줍니다.

- **ReLU** — 은닉층에서 가장 많이 씁니다. 음수는 0으로, 양수는 그대로 통과.
- **Leaky ReLU** — ReLU의 변형. 음수 구간을 완전히 0으로 죽이지 않고 살짝(예: 0.1배) 흘려보내, 뉴런이 학습을 멈추는 문제를 완화합니다.
- **Sigmoid** — 출력을 0~1로 눌러, **이진분류의 확률 출력**에 씁니다(4장 로지스틱 곡선과 같은 모양).
- **Tanh** — sigmoid와 비슷하나 출력이 **-1~1**로 중심이 0입니다. 출력이 0을 기준으로 대칭이라 은닉층에서 sigmoid보다 유리할 때가 있습니다.
- **Softmax** — 클래스가 **3개 이상인 다중분류**의 출력층에서, 각 클래스 확률을 합이 1이 되도록 냅니다(이번 모듈은 이진분류라 sigmoid 사용).

![활성화 함수 5종 비교](./imgs/module07/m7_activations.png)

::: details 더 깊이 — 어디에 무엇을 쓰나
- **은닉층**: ReLU(기본), 죽는 뉴런이 걱정되면 Leaky ReLU
- **이진분류 출력층**: Sigmoid(확률 1개)
- **다중분류 출력층**: Softmax(확률 여러 개, 합=1)
- **회귀 출력층**: 활성화 없음(선형) — 키(cm) 같은 연속값을 0~1로 누르면 안 되기 때문
:::

### 학습은 어떻게 일어나나

신경망 학습은 "예측이 정답에 가까워지도록 가중치를 조금씩 고치는" 반복입니다.

1. **손실 함수(loss)** — 예측과 정답의 차이를 하나의 숫자로 잰다. 회귀는 **MSE**(평균제곱오차), 이진분류는 **binary crossentropy**.
2. **옵티마이저(optimizer)** — 손실을 줄이는 방향으로 가중치를 갱신한다. 보통 **Adam**.
3. **에폭(epoch)** — 전체 데이터를 한 번 다 학습하는 단위. 여러 에폭 반복.
4. **배치(batch)** — 데이터를 작게 나눠(예: 32개씩) 갱신.

![경사하강법과 학습률](./imgs/module07/m7_gradient.png)

::: details 더 깊이 — 경사하강법과 학습률
손실을 가중치로 미분하면 "손실이 가장 가파르게 증가하는 방향"이 나오는데, 그 **반대 방향**으로 가중치를 조금씩(학습률만큼) 옮기는 것이 **경사하강법**입니다. 학습률이 너무 작으면 수렴이 느리고, 너무 크면 최소점을 지나쳐 **발산**할 수 있습니다(위 오른쪽 그림). Adam은 여기에 관성·적응적 학습률을 더해 더 안정적으로 수렴하게 만든 옵티마이저입니다.
:::

### 과적합과 대응

층·뉴런이 많은 신경망은 표현력이 커서, 훈련 데이터를 **외워버리는 과적합**이 나기 쉽습니다. 이 모듈에서 쓰는 세 가지 안전장치는:

- **검증 데이터(validation)** — 학습 중 별도 데이터로 성능을 지켜봐 과적합 시작을 감지
- **EarlyStopping** — 검증 손실이 더 나아지지 않으면 학습을 자동으로 멈춤
- **Dropout** — 학습 중 일부 뉴런을 무작위로 꺼서 특정 뉴런에 과의존하지 않게 함

![과적합과 EarlyStopping](./imgs/module07/m7_overfitting.png)

### 그래서 딥러닝을 언제 쓰나

딥러닝은 **이미지·음성·자연어**처럼 크고 복잡한 비정형 데이터에서 강력합니다. 반면 **이번 같은 표(tabular) 형태의 소규모 데이터**에서는, 6장에서 본 것처럼 단순한 선형모델이나 트리 기반 모델이 딥러닝만큼(때로는 더) 잘 맞고 학습도 빠릅니다.

::: warning 딥러닝이 항상 정답은 아닙니다
"신경망 = 최신 = 최고"가 아닙니다. 데이터의 크기·형태·관계 구조에 맞는 모델을 고르는 것이 핵심입니다. 이번 모듈에서 회귀·분류 각각을 6장의 머신러닝과 비교하며, **딥러닝이 언제 이점이 있고 언제 없는지**를 실제 수치로 확인합니다.
:::

## 7-2. 회귀 — height_cm 예측

6장에서 머신러닝으로 풀었던 **키(`height_cm`) 예측**을, 이번엔 신경망으로 풀고 결과를 비교합니다.

::: tip 회귀 신경망의 뼈대
- **입력** — Module 5의 전처리기(`preprocessor`)를 그대로 재사용합니다(수치 16개 특성).
- **은닉층** — `Dense(64) → Dense(32)`, 활성화 `ReLU`
- **출력층** — `Dense(1)`, **활성화 없음(선형)** — 키(cm)는 연속값이라 0~1로 누르면 안 됩니다.
- **손실** — `mse`(평균제곱오차), **옵티마이저** — `adam`
:::

```python
import numpy as np, random, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 재현성: 실행마다 결과가 흔들리지 않도록 시드를 고정합니다.
random.seed(42); np.random.seed(42); tf.random.set_seed(42)

# Module 5의 전처리기로 학습/테스트 입력을 변환합니다.
# - fit은 train에만(누수 방지), test는 transform만
X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)

# 신경망 정의 (입력 → 은닉 64 → 은닉 32 → 출력 1)
model = keras.Sequential([
    layers.Input(shape=(X_train_p.shape[1],)),   # 특성 수(16)만큼 입력
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)                              # 회귀 출력: 활성화 없음(선형)
])

# 손실=mse, 옵티마이저=adam
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# EarlyStopping: 검증 손실이 10에폭 동안 안 나아지면 멈추고 최적 가중치 복원
es = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

# 학습: 훈련 데이터의 20%를 검증용으로 떼어 과적합을 감시
history = model.fit(
    X_train_p, y_train,
    validation_split=0.2,
    epochs=100, batch_size=32,
    callbacks=[es], verbose=0
)

# 평가
pred = model.predict(X_test_p, verbose=0).flatten()
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)
print(f'MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.4f}')
```

### 학습곡선 — 과적합 감시

`history`에 저장된 훈련/검증 손실을 그려, 학습이 잘 수렴했는지 확인합니다.

```python
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('epoch'); plt.ylabel('loss (MSE)')
plt.legend(); plt.show()
```

![회귀 신경망 학습곡선](./imgs/module07/m7_reg_curve.png)

훈련·검증 손실이 함께 빠르게 내려가 낮은 값에서 수렴합니다. 두 곡선이 크게 벌어지지 않으므로 심한 과적합은 없고, EarlyStopping이 검증 손실이 가장 낮은 지점(약 23에폭)의 가중치를 복원한 뒤 33에폭에서 멈췄습니다.

### 결과 — 6장 머신러닝과 비교

![딥러닝 회귀 결과와 ML 비교](./imgs/module07/m7_reg_result.png)

| 모델 | R² | 비고 |
|---|---|---|
| **LinearRegression** | **0.8673** | 6장 최고 |
| DeepLearning (이번) | 0.8573 | 트리보다 약간 높음 |
| RandomForest | 0.8542 | 6장 |

**해석:** 딥러닝(R²=0.8573)은 RandomForest는 근소하게 앞서지만, **가장 단순한 LinearRegression(0.8673)은 넘지 못했습니다.** 이유는 6장에서 본 것과 같습니다 — 이 데이터는 `prev_height_cm`과 `height_cm`의 관계가 거의 선형이라, 복잡한 신경망보다 단순한 선형모델이 더 잘 맞습니다.

::: warning 딥러닝이 항상 이기지 않는다 — 실제 사례
표(tabular) 형태의 소규모 데이터에서는, 잘 맞는 선형·트리 모델을 신경망이 넘어서기 어려운 경우가 많습니다. 신경망은 층·뉴런·에폭 등 조정할 게 많고 학습도 느린데, 여기서는 그 비용 대비 이점이 없었습니다. **모델은 데이터 구조에 맞게 고르는 것**이 핵심입니다.
:::

## 7-3. 회귀 모델 검수 — 예측이 믿을 만한가

R²·MAE 같은 요약 숫자 하나로는 모델의 실제 행동을 다 알 수 없습니다. **예측이 어디서 잘 맞고 어디서 틀리는지**를 그림으로 확인하는 것이 검수(진단) 과정입니다. 회귀에서는 두 가지를 봅니다 — **예측값 vs 실제값**, 그리고 **잔차(residual)**.

### ① 예측값 vs 실제값

각 테스트 개체에 대해 (실제 키, 예측 키)를 점으로 찍습니다. 모든 예측이 완벽하다면 점들이 대각선 `y = x` 위에 정확히 놓입니다. 대각선에서 멀어질수록 그 개체의 예측이 크게 빗나간 것입니다.

```python
import matplotlib.pyplot as plt

plt.scatter(y_test, pred, s=10, alpha=0.35)
lim = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
plt.plot(lim, lim, '--', label='y = x (완벽 예측)')   # 기준선
plt.xlabel('실제 height_cm'); plt.ylabel('예측 height_cm')
plt.legend(); plt.show()
```

![예측값 vs 실제값](./imgs/module07/m7_reg_pred_vs_actual.png)

점들이 대체로 대각선을 따라 좁게 모여 있어, 대부분의 개체를 잘 예측합니다. 다만 두 가지 특징이 보입니다. 왼쪽 아래에 **실제 키와 무관하게 약 50cm로 예측된 수평 띠**가 있고(신경망이 특정 구간을 뭉뚱그린 흔적), 오른쪽 아래에는 **실제 220cm대를 50cm로 크게 틀린 이상치 몇 개**가 있습니다. 이런 점들이 뒤의 오차(RMSE)를 키우는 주범입니다.

### ② 잔차 분석

**잔차 = 실제값 − 예측값** 입니다. 좋은 모델의 잔차는 ⓐ **0 주변에 고르게 흩어지고**(예측값 크기와 무관하게), ⓑ **0을 중심으로 대칭인 종 모양**을 보입니다. 특정 구간에서 잔차가 한쪽으로 쏠리면, 그 구간을 모델이 체계적으로 틀리고 있다는 신호입니다.

```python
resid = y_test - pred   # 잔차

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
a1.scatter(pred, resid, s=8, alpha=0.35); a1.axhline(0, ls='--')
a1.set_xlabel('예측값'); a1.set_ylabel('잔차(실제-예측)'); a1.set_title('잔차 vs 예측값')

a2.hist(resid, bins=40); a2.axvline(0, ls='--')
a2.set_xlabel('잔차'); a2.set_title('잔차 분포')
plt.show()
```

![잔차 분석](./imgs/module07/m7_reg_residual.png)

잔차 대부분이 **0 근처에 몰려 있고 분포도 0을 중심으로 뾰족한 종 모양**이라, 전반적으로 편향 없이 잘 예측합니다. 다만 왼쪽 그림에서 **예측값 50 부근에 잔차가 세로로 크게 튀는 점들**이 보이는데, 이는 ①에서 본 "50cm 수평 띠"와 같은 개체들입니다. 잔차 평균은 약 −1.5로 0에 가깝지만, 소수의 큰 잔차가 표준편차(약 8.9)를 키웁니다.

### ③ MAE와 RMSE를 함께 읽기

- **MAE = 3.406** — 평균적으로 약 3.4cm 빗나갑니다(오차의 평균적 크기).
- **RMSE = 9.064** — MAE보다 훨씬 큽니다.

MAE와 RMSE의 **큰 차이 자체가 진단 정보**입니다. RMSE는 큰 오차를 제곱해 더 크게 반영하므로, 둘의 격차가 크다는 것은 **평소엔 잘 맞지만 가끔 크게 틀리는 개체(위의 이상치)** 가 있다는 뜻입니다. 실제로 ①·②에서 본 몇몇 큰 오차가 RMSE를 끌어올린 것입니다.

### ④ 검수 결론

- 신경망은 대부분의 개체를 편향 없이 잘 예측한다(잔차가 0 중심, 예측-실제가 대각선에 밀집).
- 소수의 큰 오차(50cm 뭉침 구간·극단 이상치)가 RMSE를 키운다.
- 그럼에도 최종 R²=0.8573으로, **6장의 LinearRegression(0.8673)을 넘지는 못한다** — 검수 그림에서도 신경망이 이 데이터에서 결정적 이점을 주지 못한 이유가 드러난다.

::: tip 검수를 왜 하나
요약 지표는 "얼마나 틀렸나"만 알려주지만, 예측-실제 그래프와 잔차는 **"어디서, 왜 틀렸나"**를 보여줍니다. 실무에서 모델을 배포하기 전 반드시 거치는 단계입니다.
:::
