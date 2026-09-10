# 1. NumPy & Pandas 기초

실제 `plant_growth.csv`(5,000개체)를 불러와서 코드와 실행 결과를 함께 봅니다.

::: warning 수치 기준
모든 수치는 원본 5,000행 기준입니다. 15행 [mini CSV](/data)로는 다른 숫자가 나옵니다.
:::

## 1-1. NumPy 기초 — Pandas의 토대

::: tip 핵심
`arr * 1.1`처럼 배열 전체에 스칼라를 곱하면 원소마다 각각 적용됩니다(브로드캐스팅). for문 없이 벡터 연산이 가능한 게 NumPy/Pandas의 핵심입니다.
:::

```python
import numpy as np

arr = np.array([12.5, 34.0, 45.9, 18.2, 60.1])  # 식물 5개 키(cm)
print('평균 키:', arr.mean())
print('표준편차:', arr.std())
print('한 달 후 예상 (10% 성장 가정):')
print(arr * 1.1)
```

![NumPy 기초](./imgs/module01/m1_numpy.png)

## 1-2. 데이터 불러오기 & 첫 탐색

```python
import pandas as pd
df = pd.read_csv('plant_growth.csv')
df.head()
```

![head](./imgs/module01/m1_head.png)

| 명령어 | 확인하는 것 |
|---|---|
| `df.head(n)` / `df.tail(n)` | 앞/뒤 n개 행 미리보기 |
| `df.shape` | (행, 열) 개수 |
| `df.info()` | 타입 + 결측치 유무 |
| `df.describe()` | 수치형 기술통계 요약 |

```python
df.info()
```

![info](./imgs/module01/m1_info.png)

::: tip 여기서 이미 결측치가 보인다
`info()`의 Non-Null Count를 보면 species, light_condition, days_since_repot, prev_height_cm에 결측치가 있다는 걸 바로 알 수 있습니다.
:::

```python
df.describe()
```

![describe](./imgs/module01/m1_describe.png)

## 1-3. 인덱싱 — loc / iloc / Boolean

| 방식 | 기준 | 예시 |
|---|---|---|
| `df.loc[]` | 라벨 기준 | `df.loc[0, 'height_cm']` |
| `df.iloc[]` | 정수 위치 기준 | `df.iloc[0, 0]` |
| Boolean Indexing | 조건식 True인 행 | `df[df['height_cm'] > 50]` |

::: warning Pandas 조건식은 if 문법과 다릅니다
여러 조건을 합칠 때 `and`/`or`가 아니라 **`&`(and) / `|`(or)**를 쓰고, 각 조건을 반드시 괄호로 감싸야 합니다.
:::

```python
# 몬스테라이면서 키 60cm 이상
tall = df[(df['species'] == '몬스테라') & (df['height_cm'] >= 60)]
```

## 1-4. 그룹화 — groupby

::: tip 핵심
`groupby().agg()`로 그룹별 통계를 한 번에 냅니다. 통합마스터가이드 3장의 "종별 평균이 다른가?"를 코드로 확인하는 셈입니다.
:::

```python
df.groupby('species')['height_cm'].agg(['mean', 'median', 'count'])
```

![groupby](./imgs/module01/m1_groupby.png)

`'·정보없음'`이 별도 그룹으로 잡히는 게 보입니다. 이건 진짜 결측치(NaN)가 아니라 **결측을 나타내려고 입력해놓은 문자열**입니다 — NaN과는 성격이 다른 "정제되지 않은 범주형 값"으로, 5장에서 NaN과 동일하게 통일시킵니다.

## 1-5. 결측치 확인 — isnull()

```python
df.isnull().sum()
```

![isnull](./imgs/module01/m1_isnull.png)

```python
# 결측치 비율(%)
(df.isnull().sum() / len(df) * 100).round(2)

# 중앙값으로 채우기 (전처리는 5장에서 본격적으로)
df['prev_height_cm'] = df['prev_height_cm'].fillna(df['prev_height_cm'].median())
```

::: warning 채우기 전에 "왜 비었는지"부터
무조건 중앙값/평균으로 채우기 전에 근본 원인(측정 안 함/값이 0/수집 오류)을 먼저 확인하세요. 고장난 온도계를 고치지 않고 숫자만 채우는 것과 같을 수 있습니다.
:::

## 1-6. 그 외 자주 쓰는 것들

```python
# 정렬
df.sort_values('height_cm', ascending=False).head()

# 새 컬럼 (벡터 연산이 apply보다 빠름)
df['growth_rate'] = (df['height_cm'] - df['prev_height_cm']) / df['prev_height_cm']
# 주의: prev_height_cm이 0이거나 결측이면 무한대/NaN 발생

# 병합 / 저장
pd.merge(df_a, df_b, on='plant_id', how='left')
df.to_csv('processed.csv', index=False)
```

## ✅ 체크리스트

- `read_csv` 후 `head/info/describe`로 첫 탐색을 할 수 있다
- `loc`/`iloc`/Boolean Indexing의 차이를 설명할 수 있다
- `groupby().agg()`로 그룹별 통계를 낼 수 있다
- `isnull().sum()`으로 결측치를 확인할 수 있다
