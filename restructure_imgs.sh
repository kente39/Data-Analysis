#!/usr/bin/env bash
# imgs/ 를 모듈별 하위 폴더로 재편하고, 마크다운의 이미지 경로를 일괄 갱신한다.
# 저장소 루트(= package.json 이 있는 폴더)에서 실행하세요:  bash restructure_imgs.sh
# Windows는 Git Bash에서 실행. 실행 전 커밋/백업 권장.
set -e

if [ ! -d docs/imgs ]; then
  echo "docs/imgs 를 찾을 수 없습니다. 저장소 루트에서 실행하세요."
  exit 1
fi

cd docs

# 1) 모듈별 폴더 생성
mkdir -p imgs/stats-intro imgs/module01 imgs/module02 imgs/module03 \
         imgs/module04 imgs/module05 imgs/module06

# 2) 접두어 기준으로 이동 (이미 옮겨졌으면 조용히 통과)
mv imgs/0[1-6]_*.png imgs/stats-intro/ 2>/dev/null || true
for n in 1 2 3 4 5 6; do
  mv imgs/m${n}_*.png imgs/module0${n}/ 2>/dev/null || true
done

# 3) 마크다운 이미지 경로 갱신 (./imgs/파일 → ./imgs/<폴더>/파일)
for f in *.md; do
  sed -i -E \
    -e 's#\./imgs/(0[1-6]_)#./imgs/stats-intro/\1#g' \
    -e 's#\./imgs/(m1_)#./imgs/module01/\1#g' \
    -e 's#\./imgs/(m2_)#./imgs/module02/\1#g' \
    -e 's#\./imgs/(m3_)#./imgs/module03/\1#g' \
    -e 's#\./imgs/(m4_)#./imgs/module04/\1#g' \
    -e 's#\./imgs/(m5_)#./imgs/module05/\1#g' \
    -e 's#\./imgs/(m6_)#./imgs/module06/\1#g' \
    "$f"
done

# 4) 검증: 하위폴더 없는 옛 경로가 남아있으면 경고
if grep -rnE '\./imgs/[0-9m][^/]*\.png' *.md >/dev/null 2>&1; then
  echo "[경고] 아직 갱신 안 된 이미지 경로가 있습니다:"
  grep -rnE '\./imgs/[0-9m][^/]*\.png' *.md
else
  echo "완료: imgs 모듈별 재편 + 경로 갱신 OK. 이제 'npm run docs:build'로 확인하세요."
fi
