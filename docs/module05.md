# 5. 전처리 — scikit-learn

머신러닝에 넣기 전, 데이터를 모델이 학습할 수 있는 형태로 다듬습니다. 결측치 채우기(SimpleImputer), 범주형 인코딩(OneHotEncoder), 스케일 맞추기(StandardScaler)와 이 셋을 묶는 Pipeline·ColumnTransformer까지 익힙니다. 여기서 만든 파이프라인은 Module 6에서 그대로 재사용합니다.

## 5-1. 가짜(텍스트형) 결측치 통일

::: tip 이 단계는 "채우기"가 아니라 "변환"입니다
`'·정보없음'`처럼 **글자로 입력된 가짜 결측치**를 진짜 결측값(`NaN`)으로 **바꾸는(변환)** 단계입니다. 빈 자리를 다른 값으로 채우는 것이 아니라, "이건 사실 빈 값"이라고 결측 표시를 한 종류로 **통일**하는 것입니다.

**왜 먼저 하나요?** 이걸 안 하면 `'·정보없음'`이 하나의 정상 카테고리처럼 취급되어, 이후 인코딩과 모델 학습이 잘못됩니다. 실제로 빈 값을 채워 넣는 처리는 5-3에서 합니다.
:::

```python
import numpy as np

# '·정보없음'은 "값이 없다"는 뜻으로 사람이 직접 적어 넣은 '텍스트형 가짜 결측치'입니다.
# 이 문자열을 진짜 결측값(NaN)으로 바꿔, 결측 표시를 한 종류(NaN)로 통일합니다.
df['species'] = df['species'].replace('·정보없음', np.nan)

# 통일 후 컬럼별 결측 개수 확인
df.isnull().sum()
```

![missing](./imgs/module05/m5_missing.png)

**결과 해석:** `species`는 원래 결측 110개에 가짜 결측 20개가 더해져 **130개**가 됩니다. 이제 모든 결측이 `NaN` 한 종류로 통일되어, 5-3의 채우기를 일관되게 적용할 수 있습니다.

## 5-2. Output / Input / 제외 컬럼

::: tip 컬럼을 세 갈래로 나눕니다
전처리에 앞서, 각 컬럼을 **예측 대상(output)** · **예측에 쓸 재료(input)** · **제외**로 나눕니다.

- **Output(타깃)** — 우리가 맞히려는 값. 이번엔 키(`height_cm`)를 예측하므로 이것이 output입니다.
- **Input(특성)** — 예측의 단서로 쓸 컬럼들. output을 뺀 "설명변수"입니다.
- **제외** — 예측에 부적절한 컬럼. 아래 기준으로 걸러냅니다.

**제외 기준:**
- **식별자** — `plant_id`는 개체 번호일 뿐 키와 인과가 없습니다(넣으면 과적합·데이터 누수 위험).
- **다른 문제의 타깃** — `is_blooming`은 이번(회귀)의 예측 대상이 아니라 분류 문제의 타깃이라 뺍니다. 결과(개화)를 원인처럼 넣으면 안 됩니다.
:::

```python
# [1] 예측 대상(output): 이번에 맞히려는 값
output_col = 'height_cm'

# [2] 예측에 쓸 입력(input): output과 제외 컬럼을 뺀 나머지
input_cols = ['species', 'pot_size', 'light_condition', 'location_type',
              'watering_per_week', 'days_since_repot', 'fertilizer_ml',
              'humidity_pct', 'temperature_c', 'prev_height_cm']

# [3] input을 자료형에 따라 다시 셋으로 나눕니다.
#     전처리 방법(스케일링/인코딩)이 자료형마다 다르기 때문입니다.
#  - 수치형: 숫자. 스케일링 대상
num_cols = ['watering_per_week', 'days_since_repot', 'fertilizer_ml',
            'humidity_pct', 'temperature_c', 'prev_height_cm']
#  - 명목형: 순서 없는 범주. OneHot 대상
nominal_cols = ['species', 'location_type']
#  - 순서형: 순서 있는 범주. Ordinal 대상
ordinal_cols = ['pot_size', 'light_condition']
```

`plant_id`는 식별자라 제외하고, `is_blooming`은 이번 회귀의 타깃이 아니므로 제외합니다. `num_cols`·`nominal_cols`·`ordinal_cols` 구분은 5-4의 인코딩과 5-6의 파이프라인에서 그대로 쓰입니다.

## 5-3. 결측치 채우기 — SimpleImputer

::: tip 이 단계는 빈 값을 "채웁니다"
5-1에서 결측을 `NaN`으로 통일했다면, 이제 그 빈 자리를 실제 값으로 **채웁니다**. 대부분의 모델은 `NaN`이 있으면 학습을 못 하기 때문입니다.

**무엇으로 채우나요?** 자료형에 따라 다릅니다.
- **수치형 → 중앙값(median)** — 평균은 이상치에 끌려가므로, 2장에서 본 오른쪽 왜도가 있는 이 데이터엔 중앙값이 안전합니다.
- **범주형 → 최빈값(most_frequent)** — 범주엔 평균/중앙값이 없으니, 가장 자주 나온 값으로 채웁니다.
:::

```python
from sklearn.impute import SimpleImputer

# 수치형: 중앙값으로 채움 (평균은 이상치·왜도에 민감하므로 median 사용)
imp_num = SimpleImputer(strategy='median')

# 범주형: 최빈값으로 채움 (범주에는 평균/중앙값이 없으므로 most_frequent)
imp_cat = SimpleImputer(strategy='most_frequent')

# fit: 채울 값(중앙값/최빈값)을 학습 → transform: 실제로 빈 자리를 채움
# 예) 수치형 컬럼에 적용
imp_num.fit(df[num_cols])
print('채운 값(각 컬럼의 중앙값):', imp_num.statistics_)
```

![imputer](./imgs/module05/m5_imputer.png)

**결과 해석:** `imputer.statistics_`에는 각 컬럼을 채우는 데 쓴 실제 값(수치형은 중앙값, 범주형은 최빈값)이 저장됩니다. 채운 뒤에는 이 값이 의도대로인지 확인하는 습관이 좋습니다.

::: warning 채우기 전에 "왜 비었는지"부터
무조건 중앙값·최빈값으로 채우기 전에, 결측의 원인(측정 안 함 / 0을 결측처럼 기록 / 수집 오류)을 먼저 살펴야 합니다. 원인에 따라 채우는 방법 자체가 달라질 수 있습니다.
:::

## 5-4. 인코딩 — 명목형 OneHot, 순서형 Ordinal

::: tip 인코딩이란?
모델은 숫자만 계산할 수 있어서, `species='몬스테라'` 같은 **글자(범주)를 숫자로 바꾸는** 작업이 필요합니다. 이것이 인코딩입니다. 단, 범주의 성격에 따라 방법이 갈립니다.

- **명목형(순서 없음) → OneHotEncoder** — `species`, `location_type`처럼 순서가 없는 범주.
- **순서형(순서 있음) → OrdinalEncoder** — `pot_size`(Small<Medium<Large), `light_condition`(Low<Medium<High)처럼 순서가 있는 범주.

**왜 명목형에 Ordinal을 쓰면 안 되나요?** 순서가 없는 `species`에 0,1,2,3…을 매기면, 모델이 "몬스테라(0) < 스투키(1)"처럼 **없는 크기 관계를 학습**합니다. 그래서 명목형은 OneHot으로 "순서 없이" 펼칩니다.
:::

::: details 두 인코더가 실제로 어떻게 바꾸나
**OneHotEncoder** — 범주마다 열을 하나씩 만들고, 해당하는 칸만 1로 표시합니다(나머지 0). 순서 정보가 생기지 않습니다.

```
location_type:  실내 → [0, 1]   (베란다=0, 실내=1)
                베란다 → [1, 0]
```
범주가 N개면 N개의 0/1 열로 펼쳐집니다.

**OrdinalEncoder** — 지정한 순서대로 정수를 매깁니다. 크기 관계가 의미 있는 순서형에만 씁니다.

```
pot_size:  Small → 0,  Medium → 1,  Large → 2
```
:::

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# 명목형: 순서 없이 범주별 0/1 열로 펼침
# - handle_unknown='ignore': 학습 때 못 본 범주가 나와도 에러 대신 전부 0 처리
# - sparse_output=False: 결과를 일반 배열로 받음(보기 쉬움)
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# 순서형: 순서를 직접 지정해 정수로 매핑
# - categories=[...]로 각 컬럼의 순서를 명시(작은 값 → 큰 값)
ord_enc = OrdinalEncoder(categories=[['Small', 'Medium', 'Large'],
                                     ['Low', 'Medium', 'High']])
```

![encoding](./imgs/module05/m5_encoding.png)

::: warning LabelEncoder를 여기 쓰지 않는 이유
`LabelEncoder`는 이름과 달리 **입력 특성(X)이 아니라 정답(y) 인코딩용**입니다. 입력 특성의 범주형에는 쓰지 않는 것이 원칙이며, 순서형은 `OrdinalEncoder(categories=...)`로 순서를 직접 지정하세요. (범주가 알파벳순으로 임의 매핑되는 부작용도 있지만 부차적 이유입니다.)
:::

## 5-5. 스케일링 — StandardScaler vs RobustScaler

::: tip 스케일링이란?
컬럼마다 **단위와 범위가 제각각**인 것을 비슷한 범위로 맞추는 작업입니다. 이 데이터만 봐도 `prev_height_cm`은 5~154, `watering_per_week`는 0~7로 스케일이 크게 다릅니다.

**왜 필요한가요?** KNN·선형회귀·신경망처럼 **거리나 가중치**로 계산하는 모델은, 값이 큰 컬럼(예: 키)이 값이 작은 컬럼(예: 물 주기)보다 부당하게 큰 영향을 줍니다. 스케일을 맞추면 모든 특성이 공평하게 반영됩니다. (트리 계열은 스케일 영향이 적습니다.)
:::

```python
from sklearn.preprocessing import StandardScaler, RobustScaler

# StandardScaler: (값 - 평균) / 표준편차 → 평균 0, 표준편차 1로 변환
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[num_cols])
```

![scaler](./imgs/module05/m5_scaler.png)

**적용 전/후 비교:** 같은 컬럼들이 스케일링 후 모두 평균≈0, 표준편차≈1로 정렬됩니다.

| 컬럼 | 전(평균) | 전(범위) | 후(평균) | 후(표준편차) |
|---|---|---|---|---|
| `prev_height_cm` | 45.8 | 5 ~ 154 | ≈0 | 1.0 |
| `watering_per_week` | 2.2 | 0 ~ 7 | ≈0 | 1.0 |
| `fertilizer_ml` | 29.5 | 0.4 ~ 150 | ≈0 | 1.0 |

원래는 컬럼마다 크기가 제각각이지만, 변환 후엔 모두 같은 기준(평균 0·표준편차 1) 위에 놓여 서로 비교 가능해집니다.

::: warning StandardScaler도 이상치 영향을 받습니다
"이상치엔 StandardScaler가 안전"은 오해입니다. StandardScaler(평균·표준편차)와 MinMaxScaler(최소·최대) **모두 이상치에 흔들립니다**. 이상치가 심하면 중앙값·IQR 기반의 `RobustScaler`가 더 적합합니다.
:::

| 스케일러 | 기준 | 이상치 영향 |
|---|---|---|
| StandardScaler | 평균·표준편차 | 받음 |
| MinMaxScaler | 최소·최대 | 가장 크게 받음 |
| RobustScaler | 중앙값·IQR | 가장 적게 받음 |

## 5-6. Pipeline + ColumnTransformer

::: tip 순서형도 스케일링까지
KNN·선형·신경망은 거리·가중치에 민감하므로 OrdinalEncoder로 바뀐 순서형(0,1,2)도 스케일링해주는 게 일관적입니다.
:::

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('scaler', StandardScaler())])
nom_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                     ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
ord_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                     ('encoder', OrdinalEncoder(categories=[['Small','Medium','Large'],['Low','Medium','High']])),
                     ('scaler', StandardScaler())])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('nom', nom_pipe, nominal_cols),
    ('ord', ord_pipe, ordinal_cols),
])
```

![pipeline](./imgs/module05/m5_pipeline.png)

::: info 버전 참고
`OneHotEncoder(sparse_output=False)`는 sklearn 최신 기준. 구버전은 `sparse=False`. (이 가이드는 1.8.0 검증)
:::

## 5-7. train_test_split — 데이터 누수 주의

::: warning fit_transform vs transform
반드시 `train_test_split` 먼저 → train에만 `fit`, test는 `transform`만. 전체에 먼저 fit하면 test 정보가 학습에 새어들어가는 **데이터 누수**가 됩니다.
:::

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_processed = preprocessor.fit_transform(X_train)  # train: fit + transform
X_test_processed = preprocessor.transform(X_test)         # test: transform만
```

원본 10개 컬럼 → 전처리 후 16개(수치형 6 + 명목형 2→8 + 순서형 2).

## ✅ 체크리스트

- 가짜 결측치를 먼저 정리했다
- 명목형 OneHot / 순서형 Ordinal을 구분하고, LabelEncoder는 y용임을 안다
- StandardScaler·MinMaxScaler·RobustScaler 차이를 안다
- Pipeline·ColumnTransformer로 한 번에 묶을 수 있다
- test에는 transform만 써야 하는 이유(데이터 누수)를 안다
