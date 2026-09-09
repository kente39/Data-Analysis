export default {
  title: 'Python 데이터분석 AtoZ',
  description: '반려식물 생육 데이터로 배우는 데이터 분석 (NumPy·Pandas부터 딥러닝까지)',
  lang: 'ko-KR',
  appearance: 'dark',
  lastUpdated: true,
  head: [
    ['link', { rel: 'preconnect', href: 'https://cdn.jsdelivr.net', crossorigin: '' }],
    ['link', {
      rel: 'stylesheet',
      as: 'style',
      crossorigin: '',
      href: 'https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css'
    }]
  ],
  themeConfig: {
    nav: [
      { text: '홈', link: '/' },
      { text: '학습 시작', link: '/module00' },
      { text: '데이터 다운로드', link: '/data' }
    ],
    sidebar: [
      {
        text: '시작하기',
        collapsed: false,
        items: [
          { text: '0. 개발환경 셋업', link: '/module00' },
          { text: '통계학 기초 입문', link: '/stats-intro' }
        ]
      },
      {
        text: 'Python 데이터 다루기',
        collapsed: false,
        items: [
          { text: '1. NumPy & Pandas 기초', link: '/module01' },
          { text: '2. 기술통계를 코드로', link: '/module02' },
          { text: '3. 시각화 (Matplotlib·Seaborn)', link: '/module03' },
          { text: '4. 통계 검정 (SciPy·statsmodels)', link: '/module04' }
        ]
      },
      {
        text: '머신러닝 & 딥러닝',
        collapsed: false,
        items: [
          { text: '5. 전처리 (scikit-learn)', link: '/module05' },
          { text: '6. 머신러닝 (scikit-learn)', link: '/module06' }
        ]
      }
    ],
    search: { provider: 'local' },
    outline: { label: '이 페이지 목차', level: [2,3] },
    docFooter: { prev: '이전', next: '다음' },
    darkModeSwitchLabel: '테마',
    returnToTopLabel: '맨 위로',
    sidebarMenuLabel: '메뉴'
  }
}
