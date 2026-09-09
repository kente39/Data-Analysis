Python 데이터분석 AtoZ - VitePress 사이트 배포 방법
=====================================================

[가장 쉬운 방법: Vercel + GitHub]

1. 이 폴더 전체를 GitHub 저장소에 올립니다.
   (node_modules 폴더는 .gitignore로 자동 제외됩니다 - 올릴 필요 없음)

2. vercel.com 접속 -> GitHub 계정으로 로그인

3. "Add New Project" -> 방금 만든 저장소 선택 -> Import

4. 설정은 vercel.json에 이미 있으므로 그대로 "Deploy" 클릭
   (혹시 수동 입력이 필요하면:
    - Build Command: npm run docs:build
    - Output Directory: docs/.vitepress/dist)

5. 1~2분 후 https://프로젝트명.vercel.app URL이 발급됩니다.
   이 URL을 디스코드에 공유하면 됩니다.

6. 이후 내용 수정: docs/*.md 파일만 고쳐서 GitHub에 push하면
   Vercel이 자동으로 다시 빌드/배포합니다.


[로컬에서 미리보기]

  npm install          (최초 1회, 인터넷 필요)
  npm run docs:dev     (개발 서버, localhost:5173)
  npm run docs:build   (배포용 빌드)
  npm run docs:preview (빌드 결과 미리보기)


[폴더 구조]
  docs/
    index.md          - 홈페이지
    stats-intro.md    - 통계 기초 입문
    module00~06.md    - 각 모듈
    data.md           - 데이터 다운로드 페이지
    public/           - 다운로드용 CSV (URL: /plant_growth.csv)
    imgs/             - 결과 이미지/차트
    .vitepress/
      config.js       - 사이트 설정(제목, 사이드바, 검색 등)

[내용 추가/수정 시]
  - 새 모듈 추가: docs/moduleNN.md 만들고, config.js의 sidebar에 링크 추가
  - 사이드바 순서/그룹: config.js의 sidebar 배열 수정
