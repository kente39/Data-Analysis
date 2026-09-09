# 실습 데이터 다운로드

이 가이드의 모든 예제는 아래 데이터로 실행됩니다.

## plant_growth.csv (전체 5,000행)

<a href="/plant_growth.csv" download>plant_growth.csv 다운로드</a>

반려식물 5,000개체의 생육 데이터입니다. 종, 화분 크기, 채광, 급수 횟수, 비료량, 이전 측정 키, 현재 키, 개화 여부 등의 컬럼으로 구성됩니다.

## plant_growth_mini15.csv (15행 샘플)

<a href="/plant_growth_mini15.csv" download>plant_growth_mini15.csv 다운로드</a>

::: warning 주의
mini CSV는 **코드 문법·구조 확인용 샘플**입니다. 가이드 본문의 수치·통계 결과는 전부 원본 5,000행 기준이므로, mini CSV로 같은 코드를 돌리면 다른 숫자가 나옵니다.
:::

## 컬럼 설명

| 컬럼 | 설명 | 타입 |
|---|---|---|
| species | 식물 종 (몬스테라/스투키/스킨답서스/고무나무/산세베리아/필로덴드론) | 범주형(명목) |
| pot_size | 화분 크기 (Small<Medium<Large) | 범주형(순서) |
| light_condition | 채광 (Low<Medium<High) | 범주형(순서) |
| location_type | 배치 장소 (실내/베란다) | 범주형(명목) |
| watering_per_week | 주간 급수 횟수 | 수치형 |
| days_since_repot | 마지막 분갈이 후 경과일 | 수치형 |
| fertilizer_ml | 비료 투입량(ml) | 수치형 |
| humidity_pct | 습도(%) | 수치형 |
| temperature_c | 온도(°C) | 수치형 |
| prev_height_cm | 이전 측정 키(cm) | 수치형 |
| height_cm | 현재 키(cm) — 회귀 타겟 | 수치형 |
| is_blooming | 개화 여부 (Y/N) — 분류 타겟 | 범주형(이진) |

::: info 데이터 출처
이 데이터는 학습용으로 절차적으로 생성한 합성(synthetic) 데이터입니다. 종별 크기 범위는 일반적인 원예 정보를 참고했으나 정밀한 실측치는 아닙니다.
:::
