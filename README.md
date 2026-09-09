# Python 데이터분석 AtoZ

반려식물 생육 데이터(`plant_growth.csv`, 5,000행)로 배우는 데이터 분석 학습 사이트.
NumPy·Pandas 기초부터 기술통계, 시각화, 통계 검정, 머신러닝, 딥러닝까지 —
모든 코드는 복붙 가능하고, 본문의 모든 수치는 실제 코드 실행값입니다.

**Live:** https://data-analysis-snowy.vercel.app/

---

## 기술 스택

- [VitePress](https://vitepress.dev/) 1.6.4 (Vue 기반 정적 사이트 생성기)
- Node.js 18 이상
- 배포: Vercel (GitHub 연동, push 시 자동 재빌드)
- 폰트: 영문 Inter + 한글 Pretendard (웹폰트 명시 로드)

---

## 구성 (모듈)

| 순서 | 페이지 | 내용 |
|---|---|---|
| — | `index.md` | 홈 |
| — | `stats-intro.md` | 통계학 기초 입문 |
| 0 | `module00.md` | 개발환경 셋업 |
| 1 | `module01.md` | NumPy & Pandas 기초 |
| 2 | `module02.md` | 기술통계를 코드로 |
| 3 | `module03.md` | 시각화 (Matplotlib·Seaborn) |
| 4 | `module04.md` | 통계 검정 (SciPy·statsmodels) |
| 5 | `module05.md` | 전처리 (scikit-learn) |
| 6 | `module06.md` | 머신러닝 (scikit-learn) |
| 7 | `module07.md` | 딥러닝 (TensorFlow·Keras) — *작업 예정* |

각 섹션은 `핵심 박스(:::tip)` → 복붙 코드블록 → 터미널 스타일 결과 이미지 순서로 구성됩니다.

---

## 폴더 구조

```
docs/
  index.md              # 홈페이지
  stats-intro.md        # 통계 기초 입문
  module00~06.md        # 각 학습 모듈
  data.md               # 데이터 다운로드 페이지
  public/               # 다운로드용 CSV (배포 URL: /plant_growth.csv)
    plant_growth.csv        # 전체 5,000행
    plant_growth_mini15.csv # 구조 확인용 15행 샘플
  imgs/                 # 결과 이미지·차트 (PNG)
  .vitepress/
    config.js           # 사이트 설정 (제목·사이드바·검색·폰트 head)
    theme/
      index.js          # 기본 테마 확장 진입점
      custom.css        # 한글 웹폰트 + 줄바꿈(keep-all) 스타일
package.json
vercel.json             # Vercel 빌드 설정
```

---

## 로컬 실행

```bash
npm install          # 최초 1회 (인터넷 필요)
npm run docs:dev     # 개발 서버 → http://localhost:5173
npm run docs:build   # 배포용 빌드 (docs/.vitepress/dist)
npm run docs:preview # 빌드 결과 미리보기
```

> `node_modules/`, `docs/.vitepress/dist/`, `docs/.vitepress/cache/`는
> `.gitignore`로 제외되므로 커밋할 필요가 없습니다.

---

## 내용 수정 / 추가

- **문서 수정:** `docs/*.md`만 고쳐서 push하면 Vercel이 자동 재빌드/배포합니다.
- **새 모듈 추가:** `docs/moduleNN.md`를 만들고, `docs/.vitepress/config.js`의
  `sidebar` 배열에 링크(`{ text: '...', link: '/moduleNN' }`)를 추가합니다.
- **사이드바 순서/그룹:** `config.js`의 `sidebar` 배열을 수정합니다.
- **결과 이미지:** `docs/imgs/`에 넣고 마크다운에서 `![alt](./imgs/파일명.png)`로 참조합니다.
  본문 수치는 반드시 원본 5,000행 실행값을 사용하고, mini CSV는 구조 확인용임을 명시합니다.

---

## 배포 (Vercel)

이미 GitHub ↔ Vercel이 연동되어 있어, `main`(또는 연결된 브랜치)에 push하면 자동 배포됩니다.
최초 연동 시 설정은 `vercel.json`에 포함되어 있습니다.

- Build Command: `npm run docs:build`
- Output Directory: `docs/.vitepress/dist`
- Framework Preset: None

---

## 데이터

`plant_growth.csv`는 학습용으로 절차적으로 생성한 합성(synthetic) 데이터입니다.
반려식물 6종(몬스테라·스투키·스킨답서스·고무나무·산세베리아·필로덴드론)의 생육 지표로 구성되며,
회귀 타깃은 `height_cm`, 분류 타깃은 `is_blooming`(Y/N)입니다.
결측치·가짜 결측(`·정보없음`)·이상치·클래스 불균형(Y 약 12%)이 의도적으로 포함되어
전처리·모델링 학습 소재로 쓰입니다.
